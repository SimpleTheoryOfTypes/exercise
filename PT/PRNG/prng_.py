import numpy as np
from typing import TypeVar, Generic, Sized, Iterable, Container, Tuple, List

class uint32_t(np.uintc):
    pass
class uint64_t(np.uint):
    pass

UINT2 = List[uint32_t]
UINT4 = List[uint32_t]

class philox_engine:
  def __init__(self, seed: uint64_t = 67280421310721, subsequence: uint64_t = 0,
               offset: uint64_t = 0, state: uint32_t = 0):
    self.kPhilox10A = 0x9E3779B9
    self.kPhilox10B = 0xBB67AE85
    self.kPhiloxSA = 0xD2511F53
    self.kPhiloxSB = 0xCD9E8D57
    self.key = [uint32_t(seed), uint32_t(uint64_t(seed) >> uint64_t(32))]
    self.counter: UINT32 = [0,0,0,0]
    self.counter[2] = uint32_t(subsequence)
    self.counter[3] = uint32_t(uint64_t(subsequence) >> uint64_t(32))
    self.state = state
    self.incr_n(offset)
    self.output: UINT4 = [0,0,0,0]

  def incr(self):
    if (self.counter[0]):
      self.counter[0] += 1
    if (self.counter[1]):
      self.counter[1] += 1
    if (self.counter[2]):
      self.counter[2] += 1

    self.counter[3] += 1

  def incr_n(self, n: uint64_t):
    nlo: uint32_t = uint32_t(n)
    nhi: uint32_t = uint32_t(uint64_t(n) >> uint64_t(32))
    self.counter[0] += nlo

    if (self.counter[0] < nlo):
      nhi += 1
      self.counter[1] += nhi
      if (nhi != 0):
          if (nhi <= self.counter[1]):
              return
    else:
      self.counter[1] += nhi
      if (nhi <= self.counter[1]):
          return

    if (self.counter[2]):
      self.counter[1] += 1
      return

    self.counter[3] += 1


  def __call__(self):
    if (self.state == 0):
      counter_: UINT4 = self.counter
      key_: UINT2 = self.key
      counter_ = self.single_round(counter_, key_)
      key_[0] += (self.kPhilox10A)
      key_[1] += (self.kPhilox10B)
      counter_ = self.single_round(counter_, key_)
      key_[0] += (self.kPhilox10A)
      key_[1] += (self.kPhilox10B)
      counter_ = self.single_round(counter_, key_)
      key_[0] += (self.kPhilox10A)
      key_[1] += (self.kPhilox10B)
      counter_ = self.single_round(counter_, key_)
      key_[0] += (self.kPhilox10A)
      key_[1] += (self.kPhilox10B)
      counter_ = self.single_round(counter_, key_)
      key_[0] += (self.kPhilox10A)
      key_[1] += (self.kPhilox10B)
      counter_ = self.single_round(counter_, key_)
      key_[0] += (self.kPhilox10A)
      key_[1] += (self.kPhilox10B)
      counter_ = self.single_round(counter_, key_)
      key_[0] += (self.kPhilox10A)
      key_[1] += (self.kPhilox10B)
      counter_ = self.single_round(counter_, key_)
      key_[0] += (self.kPhilox10A)
      key_[1] += (self.kPhilox10B)

      self.output = self.single_round(counter_, key_)
      self.incr()

    ret: uint32_t = self.output[self.state]
    self.state = np.bitwise_and(self.state + 1, uint32_t(3))
    return ret

  def mulhilo32(self, a: uint32_t, b: uint32_t, result_high: uint32_t) -> uint32_t:
    product:uint64_t = uint64_t(a) * (b)
    result_high = uint32_t(product) >> uint32_t(32)
    return uint32_t(product)

  def single_round(self, ctr: UINT4, in_key: UINT2) -> UINT4:
    hi0 = uint32_t(0xFFFFFFFF)
    hi1 = uint32_t(0xFFFFFFFF)
    lo0 = self.mulhilo32(self.kPhiloxSA, ctr[0], hi0)
    lo1 = self.mulhilo32(self.kPhiloxSB, ctr[2], hi1)
    ret = [0xFFFFFFFF, lo1, 0xFFFFFFFF, lo0]
    ret[0] = np.bitwise_xor(np.bitwise_xor(hi1, ctr[1]), in_key[0])
    ret[2] = np.bitwise_xor(np.bitwise_xor(hi0, ctr[3]), in_key[1])
    return ret

myphilox = philox_engine()

for i in range(0,10):
  print(myphilox())
