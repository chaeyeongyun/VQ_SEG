def solution(N):
  dp = [True for _ in range((10**N) + 1)]
  for i in range(2, int(((10**N) - 1)**0.5) + 1):
    print("dp")
    if dp[i]:
      j = 2
      while i * j < 10 ** N:
        dp[i * j] = False
  for i in range(10 ^ (N - 1), 10 ^ N):
    print(f"i:{i}")
    for j in range(N):
      print(f"j: {j}")
      is_break = False
      if dp[i // (10**j)]:
        break
    if not is_break:
      print(i)


solution(4)
