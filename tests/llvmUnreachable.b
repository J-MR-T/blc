// RUN: %blc -l %s 2>&1 | FileCheck %s

test(x) { register y = 55; if(x==y) { y = y + x;  return x;} return y; if(x==y) {y=y+1;return x;} return y;}
// CHECK: 55
