import domain.problem;

public abstract class optimize {
    public problem prob; //问题
    public int iterations; //最大迭代次数
    public double error; //误差阈值

    public optimize(problem prob, int iterations, double error) {
        this.prob = prob;
        this.iterations = iterations;
        this.error = error;
    }
}
