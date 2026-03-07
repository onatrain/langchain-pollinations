from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import langchain_pollinations._audio_catalog as cat

from langchain_pollinations._audio_catalog import (
    AudioModelId,
    _FALLBACK_AUDIO_MODEL_IDS,
    _audio_model_ids_lock,
    _load_audio_model_ids,
)

# Objetivo del patch para la importación local dentro de _load_audio_model_ids.
_MI_PATCH = "langchain_pollinations.models.ModelInformation"


@pytest.fixture(autouse=True)
def _reset_catalog():
    """
    Reset the mutable module-level globals to a known initial state before
    every test and restore them afterwards, ensuring full test isolation.
    """
    cat._audio_model_ids_cache = list(_FALLBACK_AUDIO_MODEL_IDS)
    cat._audio_model_ids_loaded = False
    yield
    cat._audio_model_ids_cache = list(_FALLBACK_AUDIO_MODEL_IDS)
    cat._audio_model_ids_loaded = False


def _make_mi_cls(audio_ids: list[str]) -> MagicMock:
    """
    Return a MagicMock that replaces ModelInformation and whose instance
    ``get_available_models()`` returns the supplied audio_ids.
    """
    instance = MagicMock()
    instance.get_available_models.return_value = {
        "audio": audio_ids,
        "text": [],
        "image": [],
    }
    return MagicMock(return_value=instance)


def _make_mi_cls_raising(exc: Exception) -> MagicMock:
    """Return a ModelInformation mock whose constructor raises exc."""
    return MagicMock(side_effect=exc)


def _make_mi_cls_instance_raising(exc: Exception) -> MagicMock:
    """Return a ModelInformation mock whose get_available_models raises exc."""
    instance = MagicMock()
    instance.get_available_models.side_effect = exc
    return MagicMock(return_value=instance)


class TestModuleConstants:
    def test_fallback_ids_is_list(self):
        assert isinstance(_FALLBACK_AUDIO_MODEL_IDS, list)

    def test_fallback_ids_not_empty(self):
        assert len(_FALLBACK_AUDIO_MODEL_IDS) > 0

    def test_fallback_ids_all_strings(self):
        assert all(isinstance(m, str) for m in _FALLBACK_AUDIO_MODEL_IDS)

    def test_fallback_ids_contains_tts_models(self):
        for model_id in ("openai-audio", "tts-1", "elevenmusic"):
            assert model_id in _FALLBACK_AUDIO_MODEL_IDS

    def test_fallback_ids_contains_stt_models(self):
        for model_id in ("whisper-large-v3", "whisper-1", "scribe"):
            assert model_id in _FALLBACK_AUDIO_MODEL_IDS

    def test_fallback_ids_no_duplicates(self):
        assert len(_FALLBACK_AUDIO_MODEL_IDS) == len(set(_FALLBACK_AUDIO_MODEL_IDS))

    def test_initial_cache_content_equals_fallback(self):
        assert cat._audio_model_ids_cache == list(_FALLBACK_AUDIO_MODEL_IDS)

    def test_initial_cache_is_distinct_object_from_fallback(self):
        # El módulo debe haber creado una copia con list(), no un alias.
        assert cat._audio_model_ids_cache is not _FALLBACK_AUDIO_MODEL_IDS

    def test_initial_loaded_flag_is_false(self):
        assert cat._audio_model_ids_loaded is False

    def test_loaded_flag_is_bool(self):
        assert isinstance(cat._audio_model_ids_loaded, bool)

    def test_lock_is_threading_lock_instance(self):
        assert isinstance(_audio_model_ids_lock, type(threading.Lock()))

    def test_audio_model_id_alias_is_str(self):
        assert AudioModelId is str

    def test_audio_model_id_values_are_valid_str_instances(self):
        # Verificar que el alias se puede usar para anotaciones.
        value: AudioModelId = "test-model"
        assert isinstance(value, str)


class TestReturnSemantics:
    def test_returns_list(self):
        cat._audio_model_ids_loaded = True
        result = _load_audio_model_ids()
        assert isinstance(result, list)

    def test_returns_copy_not_same_reference_as_cache(self):
        cat._audio_model_ids_loaded = True
        result = _load_audio_model_ids()
        assert result is not cat._audio_model_ids_cache

    def test_returned_list_matches_cache_content(self):
        cat._audio_model_ids_loaded = True
        snapshot = list(cat._audio_model_ids_cache)
        result = _load_audio_model_ids()
        assert result == snapshot

    def test_mutating_returned_list_does_not_alter_cache(self):
        cat._audio_model_ids_loaded = True
        snapshot = list(cat._audio_model_ids_cache)
        result = _load_audio_model_ids()
        result.append("mutant-id")
        result.clear()
        assert cat._audio_model_ids_cache == snapshot

    def test_returned_list_elements_are_strings(self):
        cat._audio_model_ids_loaded = True
        result = _load_audio_model_ids()
        assert all(isinstance(m, str) for m in result)


class TestFirstCall:
    def test_instantiates_model_information(self):
        mock_cls = _make_mi_cls(["m1"])
        with patch(_MI_PATCH, mock_cls):
            _load_audio_model_ids()
        mock_cls.assert_called_once()

    def test_calls_get_available_models_once(self):
        mock_cls = _make_mi_cls(["m1"])
        with patch(_MI_PATCH, mock_cls):
            _load_audio_model_ids()
        mock_cls.return_value.get_available_models.assert_called_once()

    def test_updates_cache_with_api_ids(self):
        api_ids = ["api-tts", "api-stt", "api-other"]
        with patch(_MI_PATCH, _make_mi_cls(api_ids)):
            _load_audio_model_ids()
        assert cat._audio_model_ids_cache == api_ids

    def test_returns_ids_from_api(self):
        api_ids = ["api-m1", "api-m2"]
        with patch(_MI_PATCH, _make_mi_cls(api_ids)):
            result = _load_audio_model_ids()
        assert result == api_ids

    def test_sets_loaded_flag_true(self):
        with patch(_MI_PATCH, _make_mi_cls(["m1"])):
            _load_audio_model_ids()
        assert cat._audio_model_ids_loaded is True

    def test_forwards_api_key_to_model_information(self):
        mock_cls = _make_mi_cls(["m1"])
        with patch(_MI_PATCH, mock_cls):
            _load_audio_model_ids(api_key="sk-test-99")
        mock_cls.assert_called_once_with(api_key="sk-test-99")

    def test_forwards_none_api_key_explicitly(self):
        mock_cls = _make_mi_cls(["m1"])
        with patch(_MI_PATCH, mock_cls):
            _load_audio_model_ids(api_key=None)
        mock_cls.assert_called_once_with(api_key=None)

    def test_default_api_key_argument_is_none(self):
        mock_cls = _make_mi_cls(["m1"])
        with patch(_MI_PATCH, mock_cls):
            _load_audio_model_ids()
        mock_cls.assert_called_once_with(api_key=None)

    def test_returned_value_is_copy_of_updated_cache(self):
        api_ids = ["x", "y", "z"]
        with patch(_MI_PATCH, _make_mi_cls(api_ids)):
            result = _load_audio_model_ids()
        assert result == api_ids
        assert result is not cat._audio_model_ids_cache


class TestAlreadyLoaded:
    def setup_method(self):
        cat._audio_model_ids_loaded = True
        cat._audio_model_ids_cache = ["cached-a", "cached-b"]

    def test_does_not_instantiate_model_information(self):
        mock_cls = _make_mi_cls(["fresh"])
        with patch(_MI_PATCH, mock_cls):
            _load_audio_model_ids()
        mock_cls.assert_not_called()

    def test_returns_cached_ids(self):
        result = _load_audio_model_ids()
        assert result == ["cached-a", "cached-b"]

    def test_cache_remains_unchanged(self):
        _load_audio_model_ids()
        assert cat._audio_model_ids_cache == ["cached-a", "cached-b"]

    def test_loaded_flag_stays_true(self):
        _load_audio_model_ids()
        assert cat._audio_model_ids_loaded is True

    def test_multiple_calls_never_hit_api(self):
        mock_cls = _make_mi_cls(["fresh"])
        with patch(_MI_PATCH, mock_cls):
            for _ in range(8):
                _load_audio_model_ids()
        mock_cls.assert_not_called()

    def test_returns_copy_not_cache_reference(self):
        result = _load_audio_model_ids()
        assert result is not cat._audio_model_ids_cache


class TestForceReload:
    def setup_method(self):
        cat._audio_model_ids_loaded = True
        cat._audio_model_ids_cache = ["stale-m"]

    def test_calls_model_information_despite_loaded_flag(self):
        mock_cls = _make_mi_cls(["fresh"])
        with patch(_MI_PATCH, mock_cls):
            _load_audio_model_ids(force=True)
        mock_cls.assert_called_once()

    def test_updates_cache_with_fresh_ids(self):
        fresh = ["fresh-1", "fresh-2"]
        with patch(_MI_PATCH, _make_mi_cls(fresh)):
            _load_audio_model_ids(force=True)
        assert cat._audio_model_ids_cache == fresh

    def test_returns_fresh_ids(self):
        fresh = ["fresh-a", "fresh-b"]
        with patch(_MI_PATCH, _make_mi_cls(fresh)):
            result = _load_audio_model_ids(force=True)
        assert result == fresh

    def test_force_on_unloaded_state_also_calls_api(self):
        cat._audio_model_ids_loaded = False
        mock_cls = _make_mi_cls(["new"])
        with patch(_MI_PATCH, mock_cls):
            _load_audio_model_ids(force=True)
        mock_cls.assert_called_once()

    def test_force_forwards_api_key(self):
        mock_cls = _make_mi_cls(["m"])
        with patch(_MI_PATCH, mock_cls):
            _load_audio_model_ids(api_key="sk-force-key", force=True)
        mock_cls.assert_called_once_with(api_key="sk-force-key")

    def test_loaded_flag_remains_true_after_force(self):
        with patch(_MI_PATCH, _make_mi_cls(["m"])):
            _load_audio_model_ids(force=True)
        assert cat._audio_model_ids_loaded is True

    def test_subsequent_call_without_force_uses_forced_result(self):
        fresh = ["force-result-1", "force-result-2"]
        with patch(_MI_PATCH, _make_mi_cls(fresh)):
            _load_audio_model_ids(force=True)
        # Segunda llamada sin force debe retornar el resultado de force.
        result = _load_audio_model_ids()
        assert result == fresh

    def test_force_called_twice_invokes_api_twice(self):
        mock_cls = _make_mi_cls(["m"])
        with patch(_MI_PATCH, mock_cls):
            _load_audio_model_ids(force=True)
            _load_audio_model_ids(force=True)
        assert mock_cls.call_count == 2


class TestApiReturnsEmptyList:
    def test_empty_response_does_not_overwrite_fallback_cache(self):
        with patch(_MI_PATCH, _make_mi_cls([])):
            _load_audio_model_ids()
        assert cat._audio_model_ids_cache == list(_FALLBACK_AUDIO_MODEL_IDS)

    def test_empty_response_returns_fallback_content(self):
        with patch(_MI_PATCH, _make_mi_cls([])):
            result = _load_audio_model_ids()
        assert result == list(_FALLBACK_AUDIO_MODEL_IDS)

    def test_empty_response_still_marks_loaded_true(self):
        with patch(_MI_PATCH, _make_mi_cls([])):
            _load_audio_model_ids()
        assert cat._audio_model_ids_loaded is True

    def test_empty_force_response_does_not_overwrite_previous_cache(self):
        cat._audio_model_ids_loaded = True
        cat._audio_model_ids_cache = ["prev-m1", "prev-m2"]
        with patch(_MI_PATCH, _make_mi_cls([])):
            _load_audio_model_ids(force=True)
        assert cat._audio_model_ids_cache == ["prev-m1", "prev-m2"]

    def test_empty_force_response_returns_previous_cache(self):
        cat._audio_model_ids_loaded = True
        cat._audio_model_ids_cache = ["prev-m"]
        with patch(_MI_PATCH, _make_mi_cls([])):
            result = _load_audio_model_ids(force=True)
        assert result == ["prev-m"]


class TestConstructorException:
    @pytest.mark.parametrize("exc", [
        ConnectionError("network down"),
        ValueError("bad api key"),
        RuntimeError("fatal"),
        Exception("generic"),
        OSError("timeout"),
    ])
    def test_exception_does_not_propagate(self, exc):
        with patch(_MI_PATCH, _make_mi_cls_raising(exc)):
            _load_audio_model_ids()  # no debe lanzar

    def test_connection_error_keeps_fallback_cache(self):
        with patch(_MI_PATCH, _make_mi_cls_raising(ConnectionError("down"))):
            _load_audio_model_ids()
        assert cat._audio_model_ids_cache == list(_FALLBACK_AUDIO_MODEL_IDS)

    def test_connection_error_returns_fallback(self):
        with patch(_MI_PATCH, _make_mi_cls_raising(ConnectionError("down"))):
            result = _load_audio_model_ids()
        assert result == list(_FALLBACK_AUDIO_MODEL_IDS)

    def test_constructor_exception_marks_loaded_true(self):
        with patch(_MI_PATCH, _make_mi_cls_raising(RuntimeError("fail"))):
            _load_audio_model_ids()
        assert cat._audio_model_ids_loaded is True

    def test_exception_after_force_preserves_previous_cache(self):
        cat._audio_model_ids_loaded = True
        cat._audio_model_ids_cache = ["keep-me"]
        with patch(_MI_PATCH, _make_mi_cls_raising(RuntimeError("fail"))):
            result = _load_audio_model_ids(force=True)
        assert cat._audio_model_ids_cache == ["keep-me"]
        assert result == ["keep-me"]


class TestGetAvailableModelsException:
    @pytest.mark.parametrize("exc", [
        RuntimeError("API down"),
        ValueError("unexpected payload"),
        Exception("critical"),
    ])
    def test_exception_does_not_propagate(self, exc):
        with patch(_MI_PATCH, _make_mi_cls_instance_raising(exc)):
            _load_audio_model_ids()  # no debe lanzar

    def test_runtime_error_keeps_fallback_cache(self):
        with patch(_MI_PATCH, _make_mi_cls_instance_raising(RuntimeError("down"))):
            _load_audio_model_ids()
        assert cat._audio_model_ids_cache == list(_FALLBACK_AUDIO_MODEL_IDS)

    def test_runtime_error_returns_fallback(self):
        with patch(_MI_PATCH, _make_mi_cls_instance_raising(RuntimeError("down"))):
            result = _load_audio_model_ids()
        assert result == list(_FALLBACK_AUDIO_MODEL_IDS)

    def test_exception_marks_loaded_true(self):
        with patch(_MI_PATCH, _make_mi_cls_instance_raising(RuntimeError("down"))):
            _load_audio_model_ids()
        assert cat._audio_model_ids_loaded is True

    def test_exception_after_force_preserves_previous_cache(self):
        cat._audio_model_ids_loaded = True
        cat._audio_model_ids_cache = ["keep-me"]
        with patch(_MI_PATCH, _make_mi_cls_instance_raising(RuntimeError("fail"))):
            result = _load_audio_model_ids(force=True)
        assert cat._audio_model_ids_cache == ["keep-me"]
        assert result == ["keep-me"]


class TestOneShotGuarantee:
    def test_api_called_exactly_once_across_many_calls(self):
        mock_cls = _make_mi_cls(["m1"])
        with patch(_MI_PATCH, mock_cls):
            for _ in range(10):
                _load_audio_model_ids()
        assert mock_cls.call_count == 1

    def test_second_call_without_force_skips_api_completely(self):
        mock_cls = _make_mi_cls(["m1"])
        with patch(_MI_PATCH, mock_cls):
            _load_audio_model_ids()
        # Segunda llamada fuera del contexto del patch: el flag True provoca
        # el retorno temprano antes de cualquier intento de importación.
        _load_audio_model_ids()
        assert mock_cls.call_count == 1

    def test_force_followed_by_no_force_hits_api_once(self):
        mock_cls = _make_mi_cls(["fresh"])
        with patch(_MI_PATCH, mock_cls):
            _load_audio_model_ids(force=True)
            _load_audio_model_ids()           # debe usar caché
        assert mock_cls.call_count == 1

    def test_n_force_calls_hit_api_n_times(self):
        mock_cls = _make_mi_cls(["m"])
        n = 5
        with patch(_MI_PATCH, mock_cls):
            for _ in range(n):
                _load_audio_model_ids(force=True)
        assert mock_cls.call_count == n

    def test_exception_on_first_call_still_marks_as_loaded(self):
        with patch(_MI_PATCH, _make_mi_cls_raising(ConnectionError("down"))):
            _load_audio_model_ids()
        assert cat._audio_model_ids_loaded is True
        # La segunda llamada NO debe volver a intentar la carga.
        mock_cls2 = _make_mi_cls(["m"])
        with patch(_MI_PATCH, mock_cls2):
            _load_audio_model_ids()
        mock_cls2.assert_not_called()


class TestThreadSafety:
    def test_concurrent_first_calls_invoke_api_exactly_once(self):
        """
        N threads synchronised at a barrier and calling _load_audio_model_ids()
        simultaneously must produce exactly one API call, guaranteed by the
        double-checked locking pattern.
        """
        n = 20
        call_counter: dict[str, int] = {"n": 0}
        barrier = threading.Barrier(n)

        def slow_get_available_models():
            # Simular latencia para maximizar la ventana de condición de carrera.
            time.sleep(0.015)
            call_counter["n"] += 1
            return {"audio": ["thread-safe-m"], "text": [], "image": []}

        instance = MagicMock()
        instance.get_available_models.side_effect = slow_get_available_models
        mock_cls = MagicMock(return_value=instance)

        errors: list[BaseException] = []

        def worker():
            try:
                barrier.wait(timeout=5.0)
                _load_audio_model_ids()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with patch(_MI_PATCH, mock_cls):
            threads = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)

        assert not errors, f"Errores en hilos: {errors}"
        assert call_counter["n"] == 1, (
            f"Se esperaba 1 llamada al API, se hicieron {call_counter['n']}"
        )

    def test_concurrent_calls_all_return_same_ids(self):
        """
        All threads must receive the same model ID list regardless of execution
        order.
        """
        n = 15
        barrier = threading.Barrier(n)
        results: list[list[str]] = []
        results_lock = threading.Lock()
        expected = ["concurrent-m1", "concurrent-m2", "concurrent-m3"]

        def worker():
            barrier.wait(timeout=5.0)
            ids = _load_audio_model_ids()
            with results_lock:
                results.append(ids)

        with patch(_MI_PATCH, _make_mi_cls(expected)):
            threads = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)

        assert len(results) == n
        for r in results:
            assert r == expected

    def test_concurrent_calls_all_return_list_copies(self):
        """
        Each thread must receive its own copy, not a shared reference to the
        cache.
        """
        n = 10
        barrier = threading.Barrier(n)
        results: list[list[str]] = []
        results_lock = threading.Lock()

        def worker():
            barrier.wait(timeout=5.0)
            ids = _load_audio_model_ids()
            with results_lock:
                results.append(ids)

        with patch(_MI_PATCH, _make_mi_cls(["shared-m"])):
            threads = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)

        # Ningún resultado debe ser el mismo objeto que el caché del módulo.
        for r in results:
            assert r is not cat._audio_model_ids_cache
        # Todos los resultados deben ser objetos distintos entre sí.
        ids_obj_ids = [id(r) for r in results]
        assert len(set(ids_obj_ids)) == n
