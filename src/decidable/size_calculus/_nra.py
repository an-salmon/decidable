"""Optional exact QF_NRA backend using the system Z3 library."""

from __future__ import annotations

from ctypes import CDLL, CFUNCTYPE, c_char_p, c_int, c_void_p
from ctypes.util import find_library


class BackendUnavailable(RuntimeError):
    pass


def solve(commands: str, names: tuple[str, ...], timeout_ms: int | None):
    library = find_library("z3")
    if library is None:
        raise BackendUnavailable("The system Z3 library is unavailable.")
    try:
        native = CDLL(library)

        def bind(name, result, *arguments):
            function = getattr(native, name)
            function.restype = result
            function.argtypes = arguments
            return function

        make_config = bind("Z3_mk_config", c_void_p)
        delete_config = bind("Z3_del_config", None, c_void_p)
        make_context = bind("Z3_mk_context", c_void_p, c_void_p)
        delete_context = bind("Z3_del_context", None, c_void_p)
        evaluate = bind("Z3_eval_smtlib2_string", c_char_p, c_void_p, c_char_p)
        handler_type = CFUNCTYPE(None, c_void_p, c_int)
        set_handler = bind("Z3_set_error_handler", None, c_void_p, handler_type)
    except (AttributeError, OSError) as error:
        raise BackendUnavailable("The system Z3 library lacks the required interface.") from error

    errors = []
    handler = handler_type(lambda _, error: errors.append(error))
    config = make_config()
    try:
        context = make_context(config)
    finally:
        delete_config(config)
    try:
        set_handler(context, handler)

        def run(script):
            answer = evaluate(context, script.encode("ascii"))
            if errors or answer is None:
                raise BackendUnavailable("The system Z3 library rejected the QF_NRA query.")
            return answer.decode("utf-8").strip()

        if timeout_ms is not None:
            run(f"(set-option :timeout {timeout_ms})")
        verdict = run(commands + "(check-sat-using qfnra-nlsat)")
        if verdict not in ("sat", "unsat", "unknown"):
            raise BackendUnavailable("The system Z3 library returned an unexpected verdict.")
        witness = run(f"(get-value ({' '.join(names)}))") if verdict == "sat" and names else None
        return verdict, witness
    finally:
        delete_context(context)
