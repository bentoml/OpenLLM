# Generated by tools/update-brew-tap.py. DO NOT EDIT!
# Please refers to the original template file Formula/openllm.rb.j2
# vim: set ft=ruby:
class Openllm < Formula
  desc "OpenLLM: Operating LLMs in production"
  homepage "https://github.com/bentoml/OpenLLM"
  version "0.4.7"
  license "Apache-2.0"
  head "https://github.com/bentoml/OpenLLM, branch: main"
  url "https://github.com/bentoml/OpenLLM/archive/v0.4.7.tar.gz"
  sha256 "74d657d7c1acd8667ba69214a0f06ea4e0c121dfa6e0e1d53684036465dc8a3c"

  on_linux do
    url "https://github.com/bentoml/OpenLLM/releases/download/v0.4.7/openllm-0.4.7-x86_64-unknown-linux-musl.tar.gz"
    sha256 "d1168d2b23078aa363fde8913c62a21a8b467800c13de5ddfaa9dc96641179f7"
  end
  on_macos do
    on_arm do
      url "https://github.com/bentoml/OpenLLM/releases/download/v0.4.7/openllm-0.4.7-aarch64-apple-darwin.tar.gz"
      sha256 "1f4387654de232ff4ed1f9bdd97dcd2a10ada566637f828fbf2214cdf88356b9"
    end
    on_intel do
      url "https://github.com/bentoml/OpenLLM/releases/download/v0.4.7/openllm-0.4.7-x86_64-apple-darwin.tar.gz"
      sha256 "407754d5d974d81c7c6cf215a466a411a20b4d2a0dfb150b180f64b6e6626bfd"
    end
  end

  def install
    on_linux do
      bin.install "openllm-0.4.7-x86_64-unknown-linux-musl" => "openllm"
    end
  on_macos do
    on_arm do
      bin.install "openllm-0.4.7-aarch64-apple-darwin" => "openllm"
    end
    on_intel do
      bin.install "openllm-0.4.7-x86_64-apple-darwin" => "openllm"
    end
  end
    ohai "To get started, run: 'openllm --help'"
    ohai "To see supported models, run: 'openllm models'"
  end

  test do
    shell_output "#{bin}/openllm --version"
  end
end
