# from models.networks.deeplabv3 import *
# from easydict import EasyDict
# if __name__ == "__main__":
#     params = EasyDict({
#             "encoder_name":"resnet50",
#             "num_classes":3,
#             "depth": 5,
#             "encoder_weights":"imagenet_swsl",
#             })
#     model = DeepLabV3(**params)
#     dummy = torch.randn(2, 3, 32, 32)
#     output = model(dummy)
#     a= 1
    
def fib(n):
        if n <= 1:
                return n
        else:
                return fib(n-1) + fib(n-2)
        
print(fib(5))

import collections
from typing import Any
class fib_memoization():
        dp = collections.defaultdict(int)
        def fib(self, n:int):
                if n <= 1:
                        return n
                if self.dp[n]:
                        return self.dp[n]
                self.dp[n] = self.fib(n-1) + self.fib(n-2)
                return self.dp[n]
        def __call__(self, n):
                return self.fib(n)
fibo = fib_memoization()
print(fibo(5))

class fib_tabulation():
        dp = collections.defaultdict(int)
        def fib(self, n:int):
                self.dp[1] = 1
                for i in range(2, n+1):
                        self.dp[i] = self.dp[i-1] + self.dp[i-2]
                return self.dp[n]
        def __call__(self, n):
                return self.fib(n)
fibo = fib_tabulation()
print(fibo(5))