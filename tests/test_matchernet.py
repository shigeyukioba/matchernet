from brica import VirtualTimeScheduler, Timing

from matchernet_py_001 import utils
from matchernet_py_001 import state
from matchernet_py_001.matchernet import Bundle, Matcher


def test_abstract_bm():
    st = state.StateMuSigma(4)
    b0 = Bundle("Bundle0", st)
    b1 = Bundle("Bundle1", st)
    b2 = Bundle("Bundle2", st)

    m01 = Matcher("Matcher01", b0, b1)
    m02 = Matcher("Matcher02", b0, b2)
    m12 = Matcher("Matcher12", b1, b2)

    s = VirtualTimeScheduler()

    bt = Timing(0, 1, 1)
    bm = Timing(1, 1, 1)

    s.add_component(b0.component, bt)
    s.add_component(b1.component, bt)
    s.add_component(b2.component, bt)

    s.add_component(m01.component, bm)
    s.add_component(m02.component, bm)
    s.add_component(m12.component, bm)

    s.step()
    s.step()
    s.step()
    s.step()

    return s


if __name__ == '__main__':
    import time

    n = 20
    start = time.time()
    # _print_level=1 # silent mode
    utils._print_level = 3  # noisy mode
    s = test_abstract_bm()
    elapsed_time = time.time() - start
    print("abstract -- elapsed_time:{0}".format(elapsed_time) + "[sec]")
