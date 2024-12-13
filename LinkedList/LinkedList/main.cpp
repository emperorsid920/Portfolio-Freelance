//
//  main.cpp
//  LinkedList
//
//  Created by Sid Kumar on 11/19/24.
//

#include <iostream>
#include <string>
using namespace std;


struct Node{
    string country;     // Country name
    string capital;     // Capital city
    long population;    // Population in millions
    double gdp;         // GDP in trillions
    Node* next;         // Pointer to the next node
};

Node* createNode(string country, string capital, long population, double gdp){
    Node* newNode = new Node();
    newNode->country = country;
    newNode->capital = capital;
    newNode->population = population;
    newNode->gdp = gdp;
    newNode->next = nullptr;
    return newNode;
}

void insertAtEnd(Node*& head, string country, string capital, long population, double gdp) {
    Node* newNode = createNode(country, capital, population, gdp);
    if (head == nullptr) { // If the list is empty
        head = newNode;
    } else {
        Node* temp = head;
        while (temp->next != nullptr) {
            temp = temp->next;
        }
        temp->next = newNode;
    }
}


void printList(Node* head) {
    Node* temp = head;
    while (temp != nullptr) {
        cout << "Country: " << temp->country << ", Capital: " << temp->capital
             << ", Population: " << temp->population << " million"
             << ", GDP: $" << temp->gdp << " trillion" << endl;
        temp = temp->next;
    }
}

void deleteNode(Node*& head, string country) {
    if (head == nullptr) return;

    // If the head node is to be deleted
    if (head->country == country) {
        Node* temp = head;
        head = head->next;
        delete temp;
        return;
    }

    Node* current = head;
    Node* prev = nullptr;

    // Traverse the list to find the node to delete
    while (current != nullptr && current->country != country) {
        prev = current;
        current = current->next;
    }

    if (current == nullptr) return; // Country not found

    prev->next = current->next;
    delete current;
}


int main(int argc, const char * argv[]) {
    // insert code here...
   
    Node* head = nullptr; // Initialize the linked list as empty

    // Insert countries with their details
    insertAtEnd(head, "Germany", "Berlin", 83, 4.0);
    insertAtEnd(head, "France", "Paris", 68, 3.5);
    insertAtEnd(head, "USA", "Washington D.C.", 331, 25.3);
    insertAtEnd(head, "UK", "London", 68, 3.1);
    insertAtEnd(head, "Mexico", "Mexico City", 126, 1.3);
    insertAtEnd(head, "Canada", "Ottawa", 38, 2.2);

    // Print the list
    cout << "Country List:" << endl;
    printList(head);

    // Delete a country
    cout << "\nDeleting France..." << endl;
    deleteNode(head, "France");

    // Print the list again
    cout << "Country List after deletion:" << endl;
    printList(head);
    
    
    return 0;
}
