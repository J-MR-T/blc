// RUN: %blc -l %s 2>&1 | FileCheck %s

// CHECK-NOT: poison
addrof(ptr) {
    auto var = 1;
    ptr[1] = &var;
    register ptr2 = &ptr[1];
    ptr2 = ptr2 + 1000;
    ptr2[0] = 2;
    return var;
}

