"""
General language utilities.
"""
import codecs


class Utf8LineWriter(object):
    """Class for writing line by line to a file.
    Takes unicode line and writes after encoding it to utf8.
    """

    def __init__(self, filename):
        """Initialize LineWriter.
        Args:
            filename: file to write to.
        """
        self.file = codecs.open(filename, 'wb', 'utf8')

    def __enter__(self):
        """For entering the context manager."""
        return self

    def writeline(self, line):
        """Write a newline char at the end."""
        self.file.write('%s\n' % line)

    def __exit__(self, ctx_type, ctx_value, ctx_traceback):
        """Close the file at the end."""
        self.file.close()
