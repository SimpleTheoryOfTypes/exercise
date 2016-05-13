int fib(int x)
{
  if (x <= 2)
      return 1;
  else
      return fib(x) + fib(x - 1);
}
