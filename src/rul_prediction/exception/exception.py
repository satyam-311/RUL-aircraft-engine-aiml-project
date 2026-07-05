"""
Custom exception handling for the RUL prediction project.

WHY: Default Python tracebacks don't tell you WHICH file/line in YOUR
pipeline failed vs. a third-party library. In production ML pipelines with
many stages (ingestion -> preprocessing -> training -> inference), a custom
exception that captures file name + line number makes debugging failures
across a long pipeline dramatically faster.

HOW: Wraps the original exception, inspects sys.exc_info() to extract the
traceback, and builds a detailed error message.

WHERE: Used throughout the project like this:

    import sys
    from rul_prediction.exception.exception import RULException

    try:
        risky_operation()
    except Exception as e:
        raise RULException(e, sys) from e
"""

import sys


class RULException(Exception):
    """Custom exception that reports the originating file and line number."""

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = str(error_message)

        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.line_number = exc_tb.tb_lineno
        else:
            self.file_name = "unknown"
            self.line_number = -1

    def __str__(self) -> str:
        return (
            f"Error occurred in script [{self.file_name}] "
            f"at line [{self.line_number}]: {self.error_message}"
        )
