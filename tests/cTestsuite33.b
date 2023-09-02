// something akin to https://github.com/c-testsuite/c-testsuite/blob/master/tests/single-exec/00033.c

// RUN: %compareBothBackends

// should return 0;

effect()
{
	register g = 1;
	return g;
}

main()
{
    register x = 0;
    
    register g = 0;
    x = 0;
    if(x && (g = effect()))
    	return 1;
    if(g)
    	return 2;
    x = 1;
    if(x && (g = effect())) {
    	if(g != 1)
    		return 3;
    } else {
    	return 4;
    }
    g = 0;
    x = 1;
    if(x || (g = effect())) {
    	if(g)
    		return 5;
    } else {
    	return 6;
    }
    x = 0;
    if(x || (g = effect())) {
    	if(g != 1)
    		return 7;
    } else {
    	return 8;
    } 
    return 0;
}

