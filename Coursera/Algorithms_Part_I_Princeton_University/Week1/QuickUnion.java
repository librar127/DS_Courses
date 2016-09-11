package algorithmPU.Week1;

/**
 * QuickUnion Implementation in Java
 */

public class QuickUnion {

	int count;
	int[] id;

	public QuickUnion(int n) {
		id = new int[n];
		for (int i = 0; i < n; i++) {
			id[i] = i;
		}
		count = n;
	}

	public int getRoot(int i) {

		while (i != id[i]) {
			i = id[i];
		}

		return i;
	}

	public boolean connected(int p, int q) {

		return getRoot(p) == getRoot(q);
	}

	public void Union(int p, int q) {
		if (getRoot(q) == getRoot(p)) {
			return;
		}

		id[getRoot(p)] = getRoot(q);
		count--;
	}

	public void printid() {

		for (int i = 0; i < id.length; i++) {
			System.out.print(id[i] + " ");
		}
		System.out.println("\n ");
	}

	public static void main(String[] args) {

		QuickUnion qu = new QuickUnion(10);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(4, 3);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(3, 8);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(6, 5);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(9, 4);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(2, 1);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(5, 0);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(7, 2);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(6, 1);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(1, 0);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(6, 7);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();
	}
}
