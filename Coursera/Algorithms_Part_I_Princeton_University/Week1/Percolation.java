package algorithmPU;

import edu.princeton.cs.algs4.WeightedQuickUnionUF;

public class Percolation {

	private WeightedQuickUnionUF weightedQuickUnionUF;
	private boolean[] isOpenList;
	private int givenN;
	private int virtualBottomIndex = 0;

	public Percolation(int N) {

		givenN = N;
		int numberOfArrayElements = N * N + 2;
		isOpenList = new boolean[numberOfArrayElements];
		virtualBottomIndex = numberOfArrayElements - 1;
		isOpenList[0] = true;
		isOpenList[virtualBottomIndex] = true;
		weightedQuickUnionUF = new WeightedQuickUnionUF(numberOfArrayElements);
	}

	public void open(int i, int j) {
		checkBoundaries(i, j);
		int index = xyToIndex(i, j);
		isOpenList[index] = true;
		connectElementToOpenNeighbors(i, j);
	}

	public boolean isOpen(int i, int j) {
		checkBoundaries(i, j);
		return isOpenList[xyToIndex(i, j)];
	}

	public boolean isFull(int i, int j) {
		checkBoundaries(i, j);
		return weightedQuickUnionUF.connected(0, xyToIndex(i, j));

	}

	public boolean percolates() {
		return weightedQuickUnionUF.connected(0, virtualBottomIndex);
	}

	private void checkBoundaries(int row, int column) {
		if (row <= 0 || row > givenN) {
			throw new IndexOutOfBoundsException("row index i out of bounds");
		}
		if (column <= 0 || column > givenN) {
			throw new IndexOutOfBoundsException("column index j out of bounds");
		}
	}

	private int xyToIndex(int row, int column) {
		return (row - 1) * givenN + column;
	}

	private void connectSiteToNeighbor(int siteIndex, int neighborRow, int neighborColumn) {
		int neighborIndex = xyToIndex(neighborRow, neighborColumn);
		weightedQuickUnionUF.union(siteIndex, neighborIndex);
	}

	private void connectElementToOpenNeighbors(int row, int column) {
		int siteIndex = xyToIndex(row, column);

		if (row == 1) {
			weightedQuickUnionUF.union(0, siteIndex);
		}

		if (row == givenN) {
			weightedQuickUnionUF.union(virtualBottomIndex, siteIndex);
		}

		if (row != 1) {
			if (isOpen(row - 1, column)) {
				connectSiteToNeighbor(siteIndex, row - 1, column);
			}
		}
		if (row != givenN) {
			if (isOpen(row + 1, column)) {
				connectSiteToNeighbor(siteIndex, row + 1, column);
			}
		}

		if (column != 1) {
			if (isOpen(row, column - 1)) {
				connectSiteToNeighbor(siteIndex, row, column - 1);
			}
		}
		if (column != givenN) {
			if (isOpen(row, column + 1)) {
				connectSiteToNeighbor(siteIndex, row, column + 1);
			}
		}
	}
}
