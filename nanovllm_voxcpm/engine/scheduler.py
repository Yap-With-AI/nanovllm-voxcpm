from collections import deque

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.sequence import Sequence, SequenceStatus
from nanovllm_voxcpm.engine.block_manager import BlockManager


class Scheduler:
    """
    Scheduler with chunked prefill support for reduced TTFB.
    
    Scheduling priority:
    1. Decode steps for running sequences (already prefilled)
    2. Continue prefill for partially-prefilled sequences
    3. Start prefill for new waiting sequences
    
    This interleaving ensures sequences that finish prefill early can
    start decoding immediately, reducing time-to-first-byte.
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.prefill_chunk_size = getattr(config, 'prefill_chunk_size', 256)
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # Sequence queues by state
        self.waiting: deque[Sequence] = deque()     # Not started
        self.prefilling: deque[Sequence] = deque()  # Mid-prefill (needs more chunks)
        self.running: deque[Sequence] = deque()     # Fully prefilled, decoding

        self._id_to_seq: dict[str, Sequence] = {}

    def is_finished(self):
        return not self.waiting and not self.prefilling and not self.running

    def add(self, seq: Sequence):
        self._id_to_seq[seq.seq_id] = seq
        self.waiting.append(seq)
    
    def cancel(self, seq_id: str):
        try:
            seq = self._id_to_seq.pop(seq_id)
        except KeyError:
            return

        self.block_manager.deallocate(seq)
        if seq.status == SequenceStatus.RUNNING:
            self.running.remove(seq)
        elif seq.status == SequenceStatus.WAITING:
            self.waiting.remove(seq)
        elif seq.status == SequenceStatus.PREFILLING:
            self.prefilling.remove(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        Schedule next batch with decode-first priority.
        
        Returns:
            tuple: (sequences, is_prefill)
        """
        # Priority 1: Decode for running sequences
        # This keeps TTFB low - sequences that finished prefill get their tokens ASAP
        if self.running:
            return self._schedule_decode()
        
        # Priority 2 & 3: Prefill (continue existing or start new)
        if self.prefilling or self.waiting:
            return self._schedule_prefill_chunk()
        
        return [], False
    
    def _schedule_prefill_chunk(self) -> tuple[list[Sequence], bool]:
        """
        Schedule prefill with chunking.
        
        Processes prefill in chunks of prefill_chunk_size tokens.
        Sequences that complete prefill move to running queue.
        """
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # Process from both queues: prefilling first (they've waited longer), then waiting
        for source in [self.prefilling, self.waiting]:
            while source and num_seqs < self.max_num_seqs:
                seq = source[0]
                
                # Calculate tokens for this chunk
                remaining_tokens = len(seq) - seq.num_cached_tokens
                if self.prefill_chunk_size > 0:
                    chunk_tokens = min(remaining_tokens, self.prefill_chunk_size)
                else:
                    # Chunking disabled - process full sequence
                    chunk_tokens = remaining_tokens
                
                if chunk_tokens <= 0:
                    # Sequence fully prefilled - shouldn't happen, but handle gracefully
                    source.popleft()
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
                    continue
                
                # Check batch limits
                if num_batched_tokens + chunk_tokens > self.max_num_batched_tokens:
                    break
                
                # Check if we can allocate blocks for this chunk
                if not self.block_manager.can_allocate(seq):
                    break
                
                # Allocate blocks if not already allocated
                if seq.status == SequenceStatus.WAITING:
                    self.block_manager.allocate(seq)
                
                source.popleft()
                seq.status = SequenceStatus.PREFILLING
                scheduled_seqs.append(seq)
                num_seqs += 1
                num_batched_tokens += chunk_tokens
        
        if scheduled_seqs:
            return scheduled_seqs, True
        
        return [], False
    
    def _schedule_decode(self) -> tuple[list[Sequence], bool]:
        """Schedule decode step for running sequences."""
        scheduled_seqs = []
        num_seqs = 0
        
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            # Check if we can append a new token
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs, False
        
        return [], False

    def after_prefill_chunk(self, seq: Sequence, tokens_processed: int):
        """
        Called after a prefill chunk completes.
        Updates sequence state and moves to appropriate queue.
        
        Args:
            seq: The sequence that completed a prefill chunk
            tokens_processed: Number of tokens processed in this chunk
        """
        # Update cached tokens count
        seq.num_cached_tokens += tokens_processed
        
        if seq.num_cached_tokens >= len(seq):
            # Prefill complete - ready for decode
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
        else:
            # More chunks needed - back to prefilling queue
            self.prefilling.append(seq)

    def preempt(self, seq: Sequence):
        """Preempt a sequence (move back to waiting)."""
        seq.status = SequenceStatus.WAITING
        seq.num_cached_tokens = 0  # Reset cache on preemption
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
    
    def finish(self, seq: Sequence):
        """Mark sequence as finished."""
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        self.running.remove(seq)
        self._id_to_seq.pop(seq.seq_id)
