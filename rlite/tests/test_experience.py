import unittest

import torch

from rlite.experience import ExperienceBatch, SimpleExperienceBatch
from rlite.utils import MPS_DEVICE, CPU_DEVICE


# noinspection DuplicatedCode
class TestExperienceBatch(unittest.TestCase):

    def test_init(self):
        batch = ExperienceBatch(
            15, (3, 2), (2, 3),
            torch.int32, torch.float32, MPS_DEVICE
        )

        self.assertTupleEqual(batch.observations.shape, (15, 3, 2))
        self.assertTupleEqual(batch.actions.shape, (15, 2, 3))
        self.assertTupleEqual(batch.rewards.shape, (15, 1))
        self.assertTupleEqual(batch.is_complete.shape, (15, 1))

        self.assertEqual(batch.observations.dtype, torch.int32)
        self.assertEqual(batch.actions.dtype, torch.float32)
        self.assertEqual(batch.rewards.dtype, torch.float32)
        self.assertEqual(batch.is_complete.dtype, torch.bool)

        self.assertTrue(batch.observations.is_mps)
        self.assertTrue(batch.actions.is_mps)
        self.assertTrue(batch.rewards.is_mps)
        self.assertTrue(batch.is_complete.is_mps)

    def test_single_append(self):
        batch = ExperienceBatch(
            15, (3, 2), (2, 3),
            torch.int32, torch.float32, MPS_DEVICE
        )

        batch.append_single_experience(
            torch.tensor([[1, 4], [8, 9], [12, 3]], dtype=torch.int32, device=MPS_DEVICE),
            torch.tensor([[3.12, 4.12, 3.321], [5.38, 7.05, 6.9]], dtype=torch.float32, device=MPS_DEVICE),
            0.98,
            True
        )

        self.assertTrue(torch.equal(
            batch.observations[0],
            torch.tensor([[1, 4], [8, 9], [12, 3]], dtype=torch.int32, device=MPS_DEVICE)
        ))

        self.assertTrue(torch.allclose(
            batch.actions[0],
            torch.tensor([[3.12, 4.12, 3.321], [5.38, 7.05, 6.9]], dtype=torch.float32, device=MPS_DEVICE)
        ))
        self.assertEqual(
            batch.rewards[0, 0],
            0.98
        )
        self.assertEqual(
            batch.is_complete[0, 0],
            True
        )

    def test_compare_multi_append(self):
        batch1 = ExperienceBatch(
            15, (3, 2), (2, 3),
            torch.float32, torch.float32, CPU_DEVICE
        )
        batch2 = ExperienceBatch(
            15, (3, 2), (2, 3),
            torch.float32, torch.float32, CPU_DEVICE
        )

        random_obs = torch.rand((50, 3, 2), dtype=torch.float32, device=CPU_DEVICE)
        random_acts = torch.rand((50, 2, 3), dtype=torch.float32, device=CPU_DEVICE)
        random_rewards = torch.rand((50, 1), dtype=torch.float32, device=CPU_DEVICE)
        random_completes = torch.randint(2, (50, 1), dtype=torch.bool, device=CPU_DEVICE)

        for i in range(50):
            batch1.append_single_experience(
                random_obs[i], random_acts[i],
                random_rewards[i, 0].item(), random_completes[i, 0].item()
            )

            batch2.append_multiple_experiences(
                random_obs[i:i+1], random_acts[i:i+1],
                random_rewards[i:i+1], random_completes[i:i+1]
            )

            self.assertTrue(torch.allclose(batch1.observations, batch2.observations))
            self.assertTrue(torch.allclose(batch1.actions, batch2.actions))
            self.assertTrue(torch.allclose(batch1.rewards, batch2.rewards))
            self.assertTrue(torch.allclose(batch1.is_complete, batch2.is_complete))

    def test_compare_multi_append_with_large_batch(self):
        batch1 = ExperienceBatch(
            15, (3, 2), (2, 3),
            torch.float32, torch.float32, CPU_DEVICE
        )


        random_obs = torch.rand((50, 3, 2), dtype=torch.float32, device=CPU_DEVICE)
        random_acts = torch.rand((50, 2, 3), dtype=torch.float32, device=CPU_DEVICE)
        random_rewards = torch.rand((50, 1), dtype=torch.float32, device=CPU_DEVICE)
        random_completes = torch.randint(2, (50, 1), dtype=torch.bool, device=CPU_DEVICE)

        for i in range(50):
            batch1.append_single_experience(
                random_obs[i], random_acts[i],
                random_rewards[i, 0].item(), random_completes[i, 0].item()
            )

            batch2 = ExperienceBatch(
                15, (3, 2), (2, 3),
                torch.float32, torch.float32, CPU_DEVICE
            )

            j = 0
            while j + 7 < i:
                batch2.append_multiple_experiences(
                    random_obs[j:j+7], random_acts[j:j+7],
                    random_rewards[j:j+7], random_completes[j:j+7]
                )
                j += 7

            batch2.append_multiple_experiences(
                random_obs[j:i+1], random_acts[j:i+1],
                random_rewards[j:i+1], random_completes[j:i+1]
            )

            self.assertTrue(torch.allclose(batch1.observations, batch2.observations))
            self.assertTrue(torch.allclose(batch1.actions, batch2.actions))
            self.assertTrue(torch.allclose(batch1.rewards, batch2.rewards))
            self.assertTrue(torch.allclose(batch1.is_complete, batch2.is_complete))

    def test_compare_multi_append_with_oversized_batches(self):
        batch1 = ExperienceBatch(
            15, (3, 2), (2, 3),
            torch.float32, torch.float32, CPU_DEVICE
        )


        random_obs = torch.rand((50, 3, 2), dtype=torch.float32, device=CPU_DEVICE)
        random_acts = torch.rand((50, 2, 3), dtype=torch.float32, device=CPU_DEVICE)
        random_rewards = torch.rand((50, 1), dtype=torch.float32, device=CPU_DEVICE)
        random_completes = torch.randint(2, (50, 1), dtype=torch.bool, device=CPU_DEVICE)

        for i in range(50):
            batch1.append_single_experience(
                random_obs[i], random_acts[i],
                random_rewards[i, 0].item(), random_completes[i, 0].item()
            )

            batch2 = ExperienceBatch(
                15, (3, 2), (2, 3),
                torch.float32, torch.float32, CPU_DEVICE
            )

            j = 0
            while j + 18 < i:
                batch2.append_multiple_experiences(
                    random_obs[j:j+18], random_acts[j:j+18],
                    random_rewards[j:j+18], random_completes[j:j+18]
                )
                j += 18

            batch2.append_multiple_experiences(
                random_obs[j:i+1], random_acts[j:i+1],
                random_rewards[j:i+1], random_completes[j:i+1]
            )

            self.assertTrue(torch.allclose(batch1.observations, batch2.observations))
            self.assertTrue(torch.allclose(batch1.actions, batch2.actions))
            self.assertTrue(torch.allclose(batch1.rewards, batch2.rewards))
            self.assertTrue(torch.allclose(batch1.is_complete, batch2.is_complete))

    def test_compare_batch_append(self):
        batch1 = ExperienceBatch(
            15, (3, 2), (2, 3),
            torch.float32, torch.float32, CPU_DEVICE
        )
        batch2 = ExperienceBatch(
            15, (3, 2), (2, 3),
            torch.float32, torch.float32, CPU_DEVICE
        )

        random_obs = torch.rand((50, 3, 2), dtype=torch.float32, device=CPU_DEVICE)
        random_acts = torch.rand((50, 2, 3), dtype=torch.float32, device=CPU_DEVICE)
        random_rewards = torch.rand((50, 1), dtype=torch.float32, device=CPU_DEVICE)
        random_completes = torch.randint(2, (50, 1), dtype=torch.bool, device=CPU_DEVICE)

        for i in range(50):
            batch1.append_single_experience(
                random_obs[i], random_acts[i],
                random_rewards[i, 0].item(), random_completes[i, 0].item()
            )

            sbe = SimpleExperienceBatch(
                random_obs[i:i+1], random_acts[i:i+1],
                random_rewards[i:i+1], random_completes[i:i+1]
            )
            batch2.append_experience_batch(sbe)

            self.assertTrue(torch.allclose(batch1.observations, batch2.observations))
            self.assertTrue(torch.allclose(batch1.actions, batch2.actions))
            self.assertTrue(torch.allclose(batch1.rewards, batch2.rewards))
            self.assertTrue(torch.allclose(batch1.is_complete, batch2.is_complete))

    def test_compare_batch_append_with_large_batch(self):
        batch1 = ExperienceBatch(
            15, (3, 2), (2, 3),
            torch.float32, torch.float32, CPU_DEVICE
        )


        random_obs = torch.rand((50, 3, 2), dtype=torch.float32, device=CPU_DEVICE)
        random_acts = torch.rand((50, 2, 3), dtype=torch.float32, device=CPU_DEVICE)
        random_rewards = torch.rand((50, 1), dtype=torch.float32, device=CPU_DEVICE)
        random_completes = torch.randint(2, (50, 1), dtype=torch.bool, device=CPU_DEVICE)

        for i in range(50):
            batch1.append_single_experience(
                random_obs[i], random_acts[i],
                random_rewards[i, 0].item(), random_completes[i, 0].item()
            )

            batch2 = ExperienceBatch(
                15, (3, 2), (2, 3),
                torch.float32, torch.float32, CPU_DEVICE
            )

            j = 0
            while j + 7 < i:
                sbe = SimpleExperienceBatch(
                    random_obs[j:j + 7], random_acts[j:j + 7],
                    random_rewards[j:j + 7], random_completes[j:j + 7]
                )
                batch2.append_experience_batch(sbe)
                j += 7

            sbe = SimpleExperienceBatch(
                random_obs[j:i + 1], random_acts[j:i + 1],
                random_rewards[j:i + 1], random_completes[j:i + 1]
            )
            batch2.append_experience_batch(sbe)

            self.assertTrue(torch.allclose(batch1.observations, batch2.observations))
            self.assertTrue(torch.allclose(batch1.actions, batch2.actions))
            self.assertTrue(torch.allclose(batch1.rewards, batch2.rewards))
            self.assertTrue(torch.allclose(batch1.is_complete, batch2.is_complete))

    def test_compare_batch_append_with_oversized_batches(self):
        batch1 = ExperienceBatch(
            15, (3, 2), (2, 3),
            torch.float32, torch.float32, CPU_DEVICE
        )


        random_obs = torch.rand((50, 3, 2), dtype=torch.float32, device=CPU_DEVICE)
        random_acts = torch.rand((50, 2, 3), dtype=torch.float32, device=CPU_DEVICE)
        random_rewards = torch.rand((50, 1), dtype=torch.float32, device=CPU_DEVICE)
        random_completes = torch.randint(2, (50, 1), dtype=torch.bool, device=CPU_DEVICE)

        for i in range(50):
            batch1.append_single_experience(
                random_obs[i], random_acts[i],
                random_rewards[i, 0].item(), random_completes[i, 0].item()
            )

            batch2 = ExperienceBatch(
                15, (3, 2), (2, 3),
                torch.float32, torch.float32, CPU_DEVICE
            )

            j = 0
            while j + 18 < i:
                sbe = SimpleExperienceBatch(
                    random_obs[j:j + 18], random_acts[j:j + 18],
                    random_rewards[j:j + 18], random_completes[j:j + 18]
                )
                batch2.append_experience_batch(sbe)
                j += 18

            sbe = SimpleExperienceBatch(
                random_obs[j:i + 1], random_acts[j:i + 1],
                random_rewards[j:i + 1], random_completes[j:i + 1]
            )
            batch2.append_experience_batch(sbe)

            self.assertTrue(torch.allclose(batch1.observations, batch2.observations))
            self.assertTrue(torch.allclose(batch1.actions, batch2.actions))
            self.assertTrue(torch.allclose(batch1.rewards, batch2.rewards))
            self.assertTrue(torch.allclose(batch1.is_complete, batch2.is_complete))




