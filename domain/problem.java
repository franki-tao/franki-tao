import Function.ht_functionN;


//问题接口
public abstract class problem {
    public ht_functionN f;
    public abstract double[] calGrad(double[] x); //对于偏导数未知情形，建议采用差分法
    public problem(ht_functionN f) {
        this.f = f;
    }
}
