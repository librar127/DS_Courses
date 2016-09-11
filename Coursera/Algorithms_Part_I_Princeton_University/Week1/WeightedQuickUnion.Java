package algorithmPU.Week1;

public class WeightedQuickUnion {

	int count;
	int id[];
	int[] sz;

	public WeightedQuickUnion(int n) {
		id = new int[n];

		for (int i = 0; i < n; i++) {
			id[i] = i;
		}
		count = n;
		sz = new int[n];
		for (int i = 0; i < n; i++) {
			sz[i] = 1;
		}

	}

	/**
	 * Find if two elements are connected
	 */
	private boolean connected(int p, int q) {

		return getRoot(p) == getRoot(q);
	}

	/**
	 * Find the Root of a an Element
	 */
	private int getRoot(int i) {

		while (i != id[i]) {
			i = id[i];
		}

		return i;
	}

	/**
	 * Union of two Elements
	 */
	private void Union(int p, int q) {

		int rootP = getRoot(p);
		int rootQ = getRoot(q);

		if (rootP == rootQ) {
			return;
		} else if (sz[rootP] < sz[rootQ]) {
			id[rootP] = rootQ;
			sz[rootQ] += sz[rootP];
		} else {
			id[rootQ] = rootP;
			sz[rootP] += sz[rootQ];
		}
		count--;

	}

	private void printSz() {

		for (int i = 0; i < id.length; i++) {
			System.out.print(sz[i] + " ");
		}
		System.out.println("\n");

	}

	private void printId() {

		for (int i = 0; i < id.length; i++) {
			System.out.print(id[i] + " ");
		}
		System.out.print("\n");

	}

	public static void main(String[] args) {

		WeightedQuickUnion wqu = new WeightedQuickUnion(10);
		System.out.println("No of connected components: " + wqu.count);
		wqu.printId();
		wqu.printSz();

		wqu.Union(2, 3);
		System.out.println("No of connected components: " + wqu.count);
		wqu.printId();
		wqu.printSz();

		wqu.Union(2, 9);
		System.out.println("No of connected components: " + wqu.count);
		wqu.printId();
		wqu.printSz();

		wqu.Union(7, 8);
		System.out.println("No of connected components: " + wqu.count);
		wqu.printId();
		wqu.printSz();

		wqu.Union(5, 6);
		System.out.println("No of connected components: " + wqu.count);
		wqu.printId();
		wqu.printSz();

		wqu.Union(5, 2);
		System.out.println("No of connected components: " + wqu.count);
		wqu.printId();
		wqu.printSz();

		wqu.Union(6, 8);
		System.out.println("No of connected components: " + wqu.count);
		wqu.printId();
		wqu.printSz();
		System.out.println(wqu.connected(2, 8));

	}
}
