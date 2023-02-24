
fn(x, y){
    register z = x;
    register a = 0;
    if((a = z*x) > y){
        return a;
    }else{
        return fn(a, y);
    }
}
