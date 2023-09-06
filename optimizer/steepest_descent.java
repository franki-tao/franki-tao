import Function.ht_functionN;
import domain.problem;

import java.util.Arrays;

//最速下降法
public class steepest_descent extends optimize{
    public double[] X0; //初始迭代点
    private double[] result;
    public steepest_descent(double[] x0, problem prob, int iterations, double error) {
        super(prob, iterations, error);
        X0 = x0;
    }


    public boolean optim(){
        double[] x = utilities.arrayCopy(X0);
        double[] x_old = utilities.arrayCopy(X0);
        double c2 = 0.9;
        double alpha;
        double[] pk;
        for (int i = 0; i < this.iterations; i++) {
            pk = utilities.minus(this.prob.calGrad(x));
            alpha = utilities.step_length(this.prob, x,1,pk,c2);
            x = utilities.add(1,x,alpha,pk);
            if(utilities.norm(x,x_old)<error) {
                result = utilities.arrayCopy(x);
                return true;
            }
            x_old = x;
        }
        return false;
    }

    //测试
    public static void main(String[] args) {
        double[] x0 = {0,0};
        problem prob = new problem(new ht_functionN() {
            @Override
            public double value(double[] params) {
                return 100*(params[1]-params[0]*params[0])*(params[1]-params[0]*params[0]) + (1-params[0])*(1-params[0]);
            }
        }) {

            @Override
            public double[] calGrad(double[] x) {
                return new double[]{200*(x[1]-x[0]*x[0])*(-2*x[0])+2*(x[0]-1), 200*(x[1]-x[0]*x[0])};
            }

        };

        steepest_descent optimize = new steepest_descent(x0, prob,1000,1e-5);
        if(optimize.optim()) {
            System.out.println(Arrays.toString(optimize.result));
        }
    }
}
