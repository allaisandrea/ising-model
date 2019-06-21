#pragma once

#include "Node.h"
#include "lattice.h"

bool Visited(Node node) { return node & 128; }

void MarkVisited(Node *node) { (*node) |= 128; }

void ClearVisitedFlag(Node *node) { (*node) &= (~128); }

void Flip(Node *node) { (*node) ^= 1; }

template <size_t nDim>
void FlipCluster(std::mt19937::result_type iProb, const Index<nDim> &i0,
                 const Index<nDim> &shape, Node *nodes, size_t *clusterSize,
                 std::mt19937 *rng, std::queue<Index<nDim>> *queue) {

    queue->emplace(i0);
    MarkVisited(nodes + GetScalarIndex(i0, shape));

    while (!queue->empty()) {
        Index<nDim> &i = queue->front();
        Node *node0 = nodes + GetScalarIndex(i, shape);
        for (size_t d = 0; d < nDim; ++d) {
            const typename Index<nDim>::value_type i_d = i[d];
            const typename Index<nDim>::value_type s_d = shape[d];
            for (size_t dir = 0; dir < 4; dir += 2) {
                i[d] = (i_d + s_d + dir - 1) % s_d;
                Node *node1 = nodes + GetScalarIndex(i, shape);
                const bool add = !Visited(*node1) && Parallel(*node0, *node1) &&
                                 (*rng)() > iProb;
                if (add) {
                    queue->emplace(i);
                    MarkVisited(node1);
                }
            }
            i[d] = i_d;
        }

        Flip(node0);
        queue->pop();
        ++(*clusterSize);
    }
}

template <size_t nDim>
void ClearVisitedFlag(const Index<nDim> &i0, const Index<nDim> &shape,
                      Node *nodes, std::queue<Index<nDim>> *queue) {

    queue->emplace(i0);
    ClearVisitedFlag(nodes + GetScalarIndex(i0, shape));

    while (!queue->empty()) {
        Index<nDim> &i = queue->front();
        for (size_t d = 0; d < nDim; ++d) {
            const typename Index<nDim>::value_type i_d = i[d];
            const typename Index<nDim>::value_type s_d = shape[d];
            for (size_t dir = 0; dir < 4; dir += 2) {
                i[d] = (i_d + s_d + dir - 1) % s_d;
                Node *node1 = nodes + GetScalarIndex(i, shape);
                if (Visited(*node1)) {
                    queue->emplace(i);
                    ClearVisitedFlag(node1);
                }
            }
            i[d] = i_d;
        }
        queue->pop();
    }
}
