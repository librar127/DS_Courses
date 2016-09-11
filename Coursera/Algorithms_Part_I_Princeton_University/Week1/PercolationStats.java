package algorithmPU;

import edu.princeton.cs.algs4.StdOut;
import edu.princeton.cs.algs4.StdRandom;
import edu.princeton.cs.algs4.StdStats;

public class PercolationStats {

	private Percolation p;
	private int[][] triedRowColumnPairs;
	private double numberOfTries = 0;
	private double[] monteCarloResults;
	private int mT;

	public PercolationStats(int N, int T) {
		if (N <= 0 || T <= 0) {
			throw new IllegalArgumentException("N, T must be bigger than 0.");
		}

		mT = T;
		monteCarloResults = new double[T];

		for (int i = 0; i < T; i++) {

			p = new Percolation(N);
			numberOfTries = 0.0;
			triedRowColumnPairs = new int[N][N];

			for (int j = 0; j < N * N * 2; j++) {
				int row = StdRandom.uniform(N) + 1;
				int column = StdRandom.uniform(N) + 1;
				if (triedRowColumnPairs[row - 1][column - 1] == 1) {
					continue;
				} else {
					triedRowColumnPairs[row - 1][column - 1] = 1;
					numberOfTries++;
					p.open(row, column);
					if (p.percolates()) {
						monteCarloResults[i] = numberOfTries / (N * N);

						break;
					}
				}
			}
		}
		p = null;
		triedRowColumnPairs = null;
	}

	public double mean() {
		return StdStats.mean(monteCarloResults);
	}

	public double stddev() {
		return StdStats.stddev(monteCarloResults);
	}

	public double confidenceLo() {
		return mean() - 1.96 * stddev() / Math.sqrt(mT);
	}

	public double confidenceHi() {
		return mean() + 1.96 * stddev() / Math.sqrt(mT);
	}

	public static void main(String[] args) {
		PercolationStats pstats = new PercolationStats(Integer.parseInt(args[0]), Integer.parseInt(args[1]));
		StdOut.println("mean\t\t\t = " + pstats.mean());
		StdOut.println("stddev\t\t\t = " + pstats.stddev());
		StdOut.println("95% confidence interval\t = " + pstats.confidenceLo() + ", " + pstats.confidenceHi());
	}
}
