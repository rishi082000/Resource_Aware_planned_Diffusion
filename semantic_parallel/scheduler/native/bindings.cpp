#include "scheduler.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using semantic_parallel::scheduler::TaskSpec;
using semantic_parallel::scheduler::WorkerAssignment;
using semantic_parallel::scheduler::schedule_round_robin;
using semantic_parallel::scheduler::schedule_lpt;
using semantic_parallel::scheduler::dynamic_claim_order;

PYBIND11_MODULE(_scheduler_native, module){
    module.doc() = "Native scheduling policies for semantic parallelism";

    py::class_<TaskSpec>(module, "TaskSpec")
    .def(
        py::init([](std::int64_t chunk_id, double estimated_cost){
            return TaskSpec{chunk_id,estimated_cost};
        }),
        py::arg("chunk_id"),
        py::arg("estimated_cost")
    )
    .def_readonly("chunk_id", &TaskSpec::chunk_id)
    .def_readonly("estimated_cost", &TaskSpec::estimated_cost);


    py::class_<WorkerAssignment>(module, "WorkerAssignment")
    .def_readonly("worker_id", &WorkerAssignment::worker_id)
    .def_readonly("chunk_ids", &WorkerAssignment::chunk_ids)
    .def_readonly("estimated_load", &WorkerAssignment::estimated_load);

    module.def(
        "schedule_round_robin",
        &schedule_round_robin,
        py::arg("tasks"),
        py::arg("worker_count")
    );

    module.def(
        "schedule_lpt",
        &schedule_lpt,
        py::arg("tasks"),
        py::arg("worker_count")
    );

    module.def(
        "dynamic_claim_order",
        &dynamic_claim_order,
        py::arg("tasks")
    );
}

