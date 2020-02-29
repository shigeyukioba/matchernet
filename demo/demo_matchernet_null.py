import numpy as np
import time
from brica import VirtualTimeScheduler, Timing
from matchernet.matchernet_null import BundleNull, MatcherNull2


def test_null():
    mu0 = np.array([0, 1, 0, 0], dtype=np.float32)
    mu1 = np.array([0, 0, 10, 0], dtype=np.float32)
    mu2 = np.array([0, 0, 0, 100], dtype=np.float32)

    b0 = BundleNull("Bundle0", mu0, delay=100)
    b1 = BundleNull("Bundle1", mu1, delay=200)
    b2 = BundleNull("Bundle2", mu2, delay=300)

    m01 = MatcherNull2("Matcher01", b0, b1)
    m02 = MatcherNull2("Matcher02", b0, b2)
    m12 = MatcherNull2("Matcher12", b1, b2)

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
    start = time.time()
    s = test_null()
    elapsed_time = time.time() - start
    print("null -- elapsed_time:{0}".format(elapsed_time) + "[sec]")
