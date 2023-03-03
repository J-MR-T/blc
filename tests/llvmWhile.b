// RUN: %blc -l %s 2>&1 | FileCheck %s

// CHECK-NOT: Warning
fn(){
    register i = 1;
    register v = calloc(2000, 1);
    register col = 0;
    while(i<2000){
    }
    while(col<4000) {
    }
    return 0;
}
