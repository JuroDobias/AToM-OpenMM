import io
import logging
import subprocess
import sys


def _test_max_level_stream_handler_filters_above_limit():
    from atom_openmm.utils.logging_filters import MaxLevelStreamHandler

    stream = io.StringIO()
    handler = MaxLevelStreamHandler(stream, "INFO")
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))

    logger = logging.getLogger("atom_openmm.test.max_level")
    old_handlers = logger.handlers[:]
    old_level = logger.level
    old_propagate = logger.propagate
    try:
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        logger.info("kept")
        logger.warning("dropped")
    finally:
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate

    assert stream.getvalue() == "INFO:kept\n"


def _test_atom_openmm_info_goes_to_stdout_and_warning_to_stderr(tmp_path):
    stdout_path = tmp_path / "stdout.txt"
    stderr_path = tmp_path / "stderr.txt"
    script = (
        "import logging, atom_openmm; "
        "logger=logging.getLogger('atom_openmm.neqti'); "
        "logger.info('routing-info'); "
        "logger.warning('routing-warning')"
    )

    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        subprocess.run([sys.executable, "-c", script], check=True, stdout=stdout, stderr=stderr)

    stdout_text = stdout_path.read_text()
    stderr_text = stderr_path.read_text()
    assert "routing-info" in stdout_text
    assert "routing-info" not in stderr_text
    assert "routing-warning" in stderr_text
    assert "routing-warning" not in stdout_text
