import ntpath, logging, sys
import datetime, jdcal, math


logger = logging.getLogger(__name__)

def path_leaf(path):
    head, tail = ntpath.split(str(path))
    return tail or ntpath.basename(head)


def positive_angle(angle):
    theta = angle
    if theta > 0:
        return theta
    elif -90 < theta < 0 :     # qudrant 4
        return theta + 360
    elif -180 < theta < -90 :
        return theta + 270
    elif -270 < theta < -180 :
        return theta + 180
    elif -360 < theta < -270 :
        return theta + 90

def quadrant(x,y):
    if x > 0 and y > 0 :     # first quadrant
        return 1
    elif x < 0 and y > 0 :   # second quadrant
        return 2
    elif x < 0 and y < 0 :
        return 3
    elif x > 0 and y < 0 :
        return 4



# current format %Y-%m-%d %H:%M:%S
def greg2jd(yyyymmddhhmmss, format="%Y-%m-%d %H:%M:%S"):

    dt = datetime.datetime.strptime(yyyymmddhhmmss, format)
    fract = 0
    if "%Y%m%d" in format:
        fract = 0
    else:
        fract = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0

    return sum(jdcal.gcal2jd(dt.year, dt.month, dt.day)) + fract


def jd2greg(jd1, jd2):
    return jdcal.jd2gcal(jd1, jd2)


def jd2hmsm(days):
    """
    Convert fractional days to hours, minutes, seconds, and microseconds.
    Precision beyond microseconds is rounded to the nearest microsecond.

    Parameters
    ----------
    days : float
        A fractional number of days. Must be less than 1.

    Returns
    -------
    hour : int
        Hour number.

    min : int
        Minute number.

    sec : int
        Second number.

    micro : int
        Microsecond number.

    Raises
    ------
    ValueError
        If `days` is >= 1.

    Examples
    --------
    >>> days_to_hmsm(0.1)
    (2, 24, 0, 0)

    """
    hours = days * 24.
    hours, hour = math.modf(hours)

    mins = hours * 60.
    mins, min = math.modf(mins)

    secs = mins * 60.
    secs, sec = math.modf(secs)

    micro = round(secs * 1.e6)


    return int(hour), int(min), int(sec), int(micro)