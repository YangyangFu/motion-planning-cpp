#include "route_planner.h"
#include <algorithm>

RoutePlanner::RoutePlanner(RouteModel &model, float start_x, float start_y, float end_x, float end_y): m_Model(model) {
    // Convert inputs to percentage:
    start_x *= 0.01;
    start_y *= 0.01;
    end_x *= 0.01;
    end_y *= 0.01;

    // Use the m_Model.FindClosestNode method to find the closest nodes to the starting and ending coordinates.
    // Store the nodes you find in the RoutePlanner's start_node and end_node attributes.
	this->start_node = &model.FindClosestNode(start_x, start_y);
  	this->end_node = &model.FindClosestNode(end_x, end_y);
}

// CalculateHValue method.
// - use the distance to the end_node for the h value.
// - Node objects have a distance method to determine the distance to another node.

float RoutePlanner::CalculateHValue(RouteModel::Node const *node) {
    return node->distance(*end_node);
}

// AddNeighbors method: expand the current node by adding all unvisited neighbors to the open list.
// - Use the FindNeighbors() method of the current_node to populate current_node.neighbors vector with all the neighbors.
// - For each node in current_node.neighbors, set the parent, the h_value, the g_value. 
// - Use CalculateHValue below to implement the h-Value calculation.
// - For each node in current_node.neighbors, add the neighbor to open_list and set the node's visited attribute to true.

// Note: g-value is (distance from node to parent node) + (distance traveled so far)  

void RoutePlanner::AddNeighbors(RouteModel::Node *current_node) {
    current_node->FindNeighbors();
    for (RouteModel::Node* node : current_node->neighbors){
        node->parent = current_node;
        node->h_value = RoutePlanner::CalculateHValue(node);
        node->g_value = current_node->distance(*node) + current_node->g_value;
        node->visited = true;
        this->open_list.push_back(node);
    }
}


// NextNode method: sort the open list and return the next node.
// - Sort the open_list according to the sum of the h value and g value.
// - Create a pointer to the node in the list with the lowest sum.
// - Remove that node from the open_list.
// - Return the pointer.

// Helper function for sort() that compares on sum h + g.
bool Compare(RouteModel::Node* node1, RouteModel::Node* node2){
    float f1 = node1->h_value + node1->g_value;
    float f2 = node2->h_value + node2->g_value;
    return f1 > f2;
}

RouteModel::Node *RoutePlanner::NextNode() {
  	RouteModel::Node* next = nullptr;

	sort(this->open_list.begin(), this->open_list.end(), 
         [](const RouteModel::Node* node1, const RouteModel::Node* node2){
      		return node1->g_value+node1->h_value < node2->g_value+node2->h_value;
    	}
    );
  	
  	next = open_list[0];
  	open_list.erase(open_list.begin());
  	
  	return next;
}

// - This method should take the current (final) node as an argument and iteratively follow the 
//   chain of parents of nodes until the starting node is found.
// - For each node in the chain, add the distance from the node to its parent to the distance variable.
// - The returned vector should be in the correct order: the start node should be the first element
//   of the vector, the end node should be the last element.

std::vector<RouteModel::Node> RoutePlanner::ConstructFinalPath(RouteModel::Node *current_node) {
    // Create path_found vector
    distance = 0.0f;
    std::vector<RouteModel::Node> path_found;
	RouteModel::Node* parent;
  
	while (current_node != this->start_node) {
      	// add to path
        path_found.emplace_back(*current_node);
      	// get parent 
      	parent = current_node->parent; 
   		// calculate distance between current node and its parent
      	distance += current_node->distance(*parent);
      	// update index
  		current_node = parent;
    } 
  	// push the start node
  	path_found.emplace_back(*current_node);
  	// reverse the path so that start node is the first element
	reverse(path_found.begin(), path_found.end());
  	  
    distance *= m_Model.MetricScale(); // Multiply the distance by the scale of the map to get meters.
    return path_found;

}


// A* Search
// - Use the AddNeighbors method to add all of the neighbors of the current node to the open_list.
// - Use the NextNode() method to sort the open_list and return the next node.
// - When the search has reached the end_node, use the ConstructFinalPath method to return the final path that was found.
// - Store the final path in the m_Model.path attribute before the method exits. This path will then be displayed on the map tile.

void RoutePlanner::AStarSearch() {
    RouteModel::Node *current_node = nullptr;
	std::vector<RouteModel::Node> final_path;
  
  	// initialize distance
  	current_node = this->start_node;
  	current_node->g_value = 0;
  	current_node->h_value = this->CalculateHValue(current_node);
	current_node->visited = true;

  	// initialize the list with starting node	
  	this->open_list.emplace_back(current_node);
  	
  	// main loop
	while (this->open_list.size() > 0) {
		// pop
		current_node = this->NextNode();
		
		// process current node
		if (current_node == this->end_node) {
			final_path = this->ConstructFinalPath(current_node);
			break;
		}

		// add neighbors
		this->AddNeighbors(current_node);
	}
  	
  	// store the final path 
  	this->m_Model.path = final_path;
  
}