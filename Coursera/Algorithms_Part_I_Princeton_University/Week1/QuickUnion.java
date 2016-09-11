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

		id[p] = getRoot(q);
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

		qu.Union(2, 3);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(2, 9);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(7, 8);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(5, 6);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();

		qu.Union(5, 2);
		System.out.println("No of connected components: " + qu.count);
		qu.printid();
	}
}
