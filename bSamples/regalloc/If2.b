
fn(x, y){
    register z = x;
    register a = 0;
    register retval = 0;
    if((a = z*x) > y){
        retval = a;
    }else{
        retval = fn(a, y);
    }
}
