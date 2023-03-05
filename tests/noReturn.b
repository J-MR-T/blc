// RUN: %blc -l %s 2>&1 | FileCheck %s

main(b){
    register a = 5;
    register retval = 0;
    if(a < b)
        retval = a*b;
    else
        retval = a/b;
    // CHECK: Warning
    // CHECK-SAME: return
    // 'forgot' the return!
}
