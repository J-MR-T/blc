// RUN: %bc -l %s 2>&1 | FileCheck %s

// CHECK-NOT: Warning
fn(x, y){
    register z = x;
    register a = 0;
    if((a = z*x) > y){
        return a;
    }else{
        return fn(a, y);
    }
}


fn2(x, y){
    register z = x;
    register a = 0;
    register retval = 0;
    if((a = z*x) > y){
        retval = a;
    }else{
        retval = fn2(a, y);
    }
}

fn3(x, y){
    auto z = x;
    auto a = 0;
    if((a = z*x) > y){
        z = 55;
        return a;
    }else{
        return fn3(a, y);
    }
}

fn4(x, y, hihi, hoho, haha){
    auto z = x;
    auto a = 0;

    if((((a = z*x || a) + z && (a && a*a)) > y)){
        if(a+hihi) a = hihi; else a = hoho;
        if(y+hoho){
            if(y+haha){

            }
            {
                if(x) return a;
            }
        }else{
            if(1) 0; else if(2) 1; else if(3){
                42;
            }else{
                9;
            }
        }
        if(a) return a; else return x;
        return a;
    }else if(5) {
        return fn4(a, y, haha, hihi, hoho);
    }else
    return 50;
}
