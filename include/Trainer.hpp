#pragma once

#include "Network.hpp"

#include <cstddef>
#include <optional>
#include <random>
#include <vector>

namespace nn {
    /*
        TRAINER:
        Drives the training loop. Shuffles the samples every epoch, slices them into mini-batches,
        runs forward -> backward on each one, and reports the epoch's mean loss.

        This matters since feeding the network the same samples in the same order through every
        epoch lets it memorize the sequence instead of the function; a fresh permutation each
        epoch breaks that spurious pattern and is the standard practice everywhere (This is exactly
        what PyTorch's DataLoader(shuffle=True) does).

        The loss returned by train_epoch() is the SAMPLE-weighted mean of the per-batch losses: each
        batch's loss contributes exactly proportional to how many samples it holds, so the reported
        value never lies about epochs with uneven last batches (One lesson I learnt from the original
        version).

        A fixed seed makes runs reproducible: same data, same seed, same trajectory. Without one, a
        non-deterministic std::random_device is used.
    */
    class Trainer {
        public:
            /*
                TRAINER CONSTRUCTOR:
                Binds to a network (By reference; the trainer does not own it) and fixes the
                batch size. Note that batch_size must be positive.
            */
            Trainer(Network & network, std::size_t batch_size,
                    std::optional<std::uint32_t> shuffle_seed = std::nullopt);

            /*
                TRAIN EPOCH:
                One full pass over the data: shuffle, slice into mini-batches, train each.
                Returns the epoch's sample-weighted mean loss.
            */
            [[nodiscard]] double train_epoch(const Eigen::MatrixXd & samples,
                                             const Eigen::MatrixXd & targets);

            /*
                FIT:
                Convenience loop. Runs train_epoch() the given number of times and returns the
                loss after every epoch, so callers can plot or log progress.
            */
            [[nodiscard]] std::vector<double> fit(const Eigen::MatrixXd & samples,
                                                  const Eigen::MatrixXd & targets, int epochs);

            // A simple batch size getter.
            [[nodiscard]] std::size_t batch_size() const { return batch_size_; }

        private:
            /*
                INNER VARIABLES:
                The network is referenced, not owned; the RNG persists across epochs so every
                epoch gets a different (But reproducible) permutation.
            */
            Network & network_;
            std::size_t batch_size_;
            std::mt19937 rng_;
    }; // class Trainer
} // namespace nn
