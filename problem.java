package ht_Math.Solver.domain;

import ht_Math.Function.ht_functionN;
import ht_Math.Matrix.h_matrix;

//问题接口
public abstract class problem {
    public ht_functionN f;
    public abstract double[] calGrad(double[] x); //对于偏导数未知情形，建议采用差分法
    public abstract h_matrix calHess(double[] x);
    public problem(ht_functionN f) {
        this.f = f;
    }
}
