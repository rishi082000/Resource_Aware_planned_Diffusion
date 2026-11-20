#pragma once

#include <cstdint>
#include <vector>

namespace semantic_parallel::scheduler {

    struct TaskSpec {
        std::int64_t chunk_id;
        double estimated_cost;
    };

    struct WorkerAssignment {
        std::int64_t worker_id;
        std::vector<std::int64_t> chunk_ids;
        double estimated_load;
    };

    std::vector<WorkerAssignment> schedule_round_robin(
        const std::vector<TaskSpec>& tasks,
        std::int64_t worker_count
    );

    std::vector<WorkerAssignment> schedule_lpt(
        const std::vector<TaskSpec>& tasks,
        std::int64_t worker_count
    );

    // Order in which workers claim chunks from the shared queue: largest estimated cost first
    std::vector<std::int64_t> dynamic_claim_order(
        const std::vector<TaskSpec>& tasks
    );
}
