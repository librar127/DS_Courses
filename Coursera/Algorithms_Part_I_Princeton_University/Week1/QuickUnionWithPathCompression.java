package algorithmPU.Week1;

public class QuickUnionWithPathCompression {

	int count;
	int id[];

	public QuickUnionWithPathCompression(int n) {
		id = new int[n];

		for (int i = 0; i < n; i++) {
			id[i] = i;
		}
		count = n;
	}

	private boolean connected(int p, int q) {
		return getRoot(p) == getRoot(q);
	}

	private int getRoot(int p) {

		while (p != id[p]) {
			id[p] = id[id[p]];
			p = id[p];
		}

		return p;
	}

	private void Union(int p, int q) {

		int rootP = getRoot(p);
		int rootQ = getRoot(q);

		if (rootP == rootQ) {
			return;
		}

		id[rootP] = rootQ;
		count--;

	}

	private void printId() {

		for (int i = 0; i < id.length; i++) {
			System.out.print(id[i] + " ");
		}
		System.out.println("\n");

	}

	public static void main(String[] args) {

		QuickUnionWithPathCompression qu = new QuickUnionWithPathCompression(10);
		System.out.println("No of connected components: " + qu.count);
		qu.printId();

		System.out.println("No of connected components: " + qu.count);
		qu.printId();

		qu.Union(4, 3);
		System.out.println("No of connected components: " + qu.count);
		qu.printId();

		qu.Union(3, 8);
		System.out.println("No of connected components: " + qu.count);
		qu.printId();

		qu.Union(6, 5);
		System.out.println("No of connected components: " + qu.count);
		qu.printId();

		qu.Union(9, 4);
		System.out.println("No of connected components: " + qu.count);
		qu.printId();

		qu.Union(2, 1);
		System.out.println("No of connected components: " + qu.count);
		qu.printId();

		qu.Union(5, 0);
		System.out.println("No of connected components: " + qu.count);
		qu.printId();

		qu.Union(7, 2);
		System.out.println("No of connected components: " + qu.count);
		qu.printId();

		qu.Union(6, 1);
		System.out.println("No of connected components: " + qu.count);
		qu.printId();

		qu.Union(1, 0);
		System.out.println("No of connected components: " + qu.count);
		qu.printId();

		System.out.println(qu.connected(2, 8));
	}

}
