#
# Eric Jeschke  (eric@naoj.org)
#
import os
import re
import Bunch


frame_regex1 = re.compile('^(\w{3})([AaQq])(\d{8})$')
frame_regex2 = re.compile('^(\w{3})([AaQq])(\d{1})(\d{7})$')
frame_templ = "%3.3s%1.1s%1.1s%07d"

class FitsFrameIdError(Exception):
    pass

class Frame(object):
    """
    Class to extract Subaru telescope frame information from a file
    path which contains a name conforming to the Subaru telescope naming
    conventions.

    Usage:
        frame = Frame('/path/to/some/file.fits')

    """

    def __init__(self, path=None):
        self.path = None
        self.filename = None
        self.directory = None
        self.inscode = None
        self.frametype = None
        self.prefix = None
        self.number = None

        if path != None:
            self.create_from_path(path)

    def create_from_path(self, path):

        # Extract frame id from file path
        (fitsdir, filename) = os.path.split(path)
        (frameid, ext) = os.path.splitext(filename)

        match = frame_regex2.match(frameid)
        if not match:
            raise ValueError("Filename (%s) does not match frame spec" % (
                filename))

        (inscode, frametype, framepfx, frame_no) = match.groups()

        self.path = path
        self.filename = filename
        self.directory = fitsdir
        self.inscode = inscode.upper()
        self.frametype = frametype.upper()
        self.prefix = str(framepfx)
        self.number = int(frame_no)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return frame_templ % (self.inscode, self.frametype, self.prefix,
                              self.number)


# OLD STYLE
def getFrameInfoFromPath(fitspath):
    """
    DO NOT USE--TO BE DEPRECATED
    Function to extract Subaru telescope frame information from a file
    path which contains a name conforming to the Subaru telescope naming
    conventions.
    """
    # Extract frame id from file path
    (fitsdir, fitsname) = os.path.split(fitspath)
    (frameid, ext) = os.path.splitext(fitsname)

    match = frame_regex1.match(frameid)
    if match:
        (inscode, frametype, frame_no) = match.groups()
        frame_no = int(frame_no)

        frameid = frameid.upper()
        inscode = inscode.upper()

        return Bunch.Bunch(frameid=frameid, fitsname=fitsname,
                           fitsdir=fitsdir, inscode=inscode,
                           frametype=frametype, frame_no=frame_no)

    raise FitsFrameIdError("path does not match Subaru FITS specification: '%s'" % (
            fitspath))

#END
