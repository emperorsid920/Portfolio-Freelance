#include <iostream>
#include <vector>
using namespace std;

// Constants for grid size and the number of days until the original infection recovers
const int SIZE = 7;
const int RECOVERY_DAYS = 15;

// Directions for adjacent cells (up, down, left, right)
int dx[] = {0, 0, -1, 1};
int dy[] = {-1, 1, 0, 0};

// Function declarations
void printGrid(const vector<vector<char>>&, int);
bool inBounds(int, int);
void updateGrid(vector<vector<char>>&, vector<vector<int>>&, int);
bool allRecovered(const vector<vector<char>>&);

int main() {
    // Create a 2D grid initialized with 'S' (Susceptible)
    vector<vector<char>> grid(SIZE, vector<char>(SIZE, 'S'));

    // Create a 2D grid to track infection days, initialized to -1 (not infected)
    vector<vector<int>> infectionDays(SIZE, vector<int>(SIZE, -1));

    int startX, startY;

    // Prompt user for the coordinates of the initial infected individual
    cout << "Enter initial infected coordinates (x y): ";
    cin >> startX >> startY;

    // Mark the initial infected cell in the grid
    grid[startX][startY] = 'I';
    infectionDays[startX][startY] = 0; // Day 0 of infection

    int day = 0; // Initialize the day counter
    printGrid(grid, day); // Print the initial state of the grid

    // Continue the simulation until all cells are 'R' (Recovered)
    while (!allRecovered(grid)) {
        // Update the grid state and infection days for the next day
        updateGrid(grid, infectionDays, day);

        // Increment the day counter
        day++;

        // Print the updated state of the grid
        printGrid(grid, day);
    }

    // Print the day when all cells have recovered
    cout << "All individuals have recovered by Day " << day << "!" << endl;

    return 0;
}

// Function to print the current state of the grid
void printGrid(const vector<vector<char>>& grid, int day) {
    cout << "Day " << day << endl;

    // Loop through each row of the grid
    for (int i = 0; i < SIZE; i++) {
        // Loop through each column in the current row
        for (int j = 0; j < SIZE; j++) {
            cout << grid[i][j] << " "; // Print the cell's state (S, I, or R)
        }
        cout << endl; // Move to the next line after each row
    }
    cout << endl; // Add an extra line for separation between days
}

// Function to check if a cell's coordinates are within the grid boundaries
bool inBounds(int x, int y) {
    return x >= 0 && x < SIZE && y >= 0 && y < SIZE;
}

// Function to update the grid for the next day
void updateGrid(vector<vector<char>>& grid, vector<vector<int>>& infectionDays, int day) {
    // Create a new grid to hold the updated state of cells
    vector<vector<char>> newGrid = grid;

    // Loop through each cell in the grid
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            // If the cell is susceptible ('S')
            if (grid[i][j] == 'S') {
                // Check all 4 adjacent cells for infection ('I')
                for (int d = 0; d < 4; d++) {
                    int ni = i + dx[d]; // Calculate the row of the adjacent cell
                    int nj = j + dy[d]; // Calculate the column of the adjacent cell

                    // If the adjacent cell is within bounds and is infected
                    if (inBounds(ni, nj) && grid[ni][nj] == 'I') {
                        newGrid[i][j] = 'I'; // Infect this cell
                        infectionDays[i][j] = day + 1; // Start tracking its infection day
                        break; // Exit the loop since infection only needs one adjacent 'I'
                    }
                }
            }
            // If the cell is infected ('I')
            else if (grid[i][j] == 'I') {
                // If this is the original infected cell and 10 days have passed
                if (infectionDays[i][j] == 0 && day == RECOVERY_DAYS) {
                    newGrid[i][j] = 'R'; // Recover this cell
                } else if (infectionDays[i][j] > 0) {
                    // Check all 4 adjacent cells for recovery ('R')
                    for (int d = 0; d < 4; d++) {
                        int ni = i + dx[d]; // Calculate the row of the adjacent cell
                        int nj = j + dy[d]; // Calculate the column of the adjacent cell

                        // If the adjacent cell is within bounds and recovered
                        if (inBounds(ni, nj) && grid[ni][nj] == 'R') {
                            newGrid[i][j] = 'R'; // Recover this cell
                            break; // Exit the loop since recovery only needs one adjacent 'R'
                        }
                    }
                }
            }
        }
    }

    // Update the main grid to the new state
    grid = newGrid;
}

// Function to check if all cells in the grid are 'R' (Recovered)
bool allRecovered(const vector<vector<char>>& grid) {
    // Loop through each cell in the grid
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (grid[i][j] != 'R') return false; // If any cell is not 'R', return false
        }
    }
    return true; // If all cells are 'R', return true
}

