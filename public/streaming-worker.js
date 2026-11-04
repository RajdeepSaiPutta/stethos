// streaming-worker.js - AudioWorkletProcessor for real-time breath sound streaming
// Runs in AudioWorklet for low-latency audio processing

class StreamingProcessor extends AudioWorkletProcessor {
  constructor(options = {}) {
    super();
    this.windowSize = options.processorOptions?.windowSize || 32000; // 2s at 16kHz
    this.hopSize = options.processorOptions?.hopSize || 16000; // 1s at 16kHz
    this.sampleRate = options.processorOptions?.sampleRate || 16000;

    this.buffer = new Float32Array(this.windowSize * 2);
    this.writePos = 0;
    this.readPos = 0;
    this.windowCount = 0;

    this.port.onmessage = (e) => {
      if (e.data.type === 'config') {
        this.windowSize = e.data.windowSize || this.windowSize;
        this.hopSize = e.data.hopSize || this.hopSize;
        this.sampleRate = e.data.sampleRate || this.sampleRate;
        this.buffer = new Float32Array(this.windowSize * 2);
        this.writePos = 0;
        this.readPos = 0;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length > 0 && input[0].length > 0) {
      const channel = input[0];

      // Write to ring buffer
      for (let i = 0; i < channel.length; i++) {
        this.buffer[this.writePos] = channel[i];
        this.writePos = (this.writePos + 1) % this.buffer.length;
      }

      // Check if we have enough for a window
      const available = (this.writePos - this.readPos + this.buffer.length) % this.buffer.length;
      if (available >= this.windowSize) {
        // Extract window
        const window = new Float32Array(this.windowSize);
        for (let i = 0; i < this.windowSize; i++) {
          window[i] = this.buffer[(this.readPos + i) % this.buffer.length];
        }

        // Send to main thread for inference
        this.port.postMessage({
          type: 'window',
          data: window,
          windowIndex: this.windowCount++,
          timestamp: currentTime
        });

        // Advance by hop
        this.readPos = (this.readPos + this.hopSize) % this.buffer.length;
      }
    }
    return true;
  }
}

registerProcessor('streaming-processor', StreamingProcessor);