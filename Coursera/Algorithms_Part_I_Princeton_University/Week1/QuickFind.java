package algorithmPU.Week1;

public class QuickFind {

	int[] id;
	int count;

	QuickFind(int n) {

		id = new int[n];
		for (int i = 0; i < n; i++) {
			id[i] = i;
		}
		count = n;
	}

	/**
	 * Find if two components are connected
	 */
	public boolean connected(int p, int q) {

		return id[p] == id[q];
	}

	/**
	 * Union between two components
	 */
	public void union(int p, int q) {

		int i = id[p];
		int j = id[q];

		for (int k = 0; k < id.length; k++) {
			if (id[k] == i) {
				id[k] = j;
			}
		}
		count--;

	}

	public void printID() {
		for (int i = 0; i < id.length; i++) {
			System.out.print(id[i] + " ");
		}
		System.out.println("\n");
	}

	public static void main(String[] args) {

		QuickFind qu = new QuickFind(10);
		qu.printID();

		qu.union(5, 2);
		System.out.println("No of Connected Components: " + qu.count);
		qu.printID();

		qu.union(3, 2);
		System.out.println("No of Connected Components: " + qu.count);
		qu.printID();

	}

}
