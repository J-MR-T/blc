// RUN: %blc %s 2>&1 | FileCheck %s

// CHECK: Warning
// CHECK-SAME: poison
foo(a, haha, hihi){
    return a - (1 << 64) >> 64 + haha* hihi % haha-1;
}
