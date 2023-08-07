# vim: set ft=ruby:
class Openllm < Formula
  desc "OpenLLM: Operating LLMs in production"
  homepage "https://github.com/bentoml/OpenLLM"
  license "Apache-2.0"
  version "0.2.16"
  head "https://github.com/bentoml/OpenLLM.git, branch: main"
  url "https://github.com/bentoml/OpenLLM/archive/v.tar.gz"
  sha256 "0ec5d8f48565b07193446ff68bb11cb22722b8a30578af9b455aa73bb67542c0"

  def install
  end

  test do
    # `test do` will create, run in and delete a temporary directory.
    #
    # This test will fail and we won't accept that! For Homebrew/homebrew-core
    # this will need to be a test that verifies the functionality of the
    # software. Run the test with `brew test openllm`. Options passed
    # to `brew install` such as `--HEAD` also need to be provided to `brew test`.
    #
    # The installed folder is not in the path, so use the entire path to any
    # executables being tested: `system "#{bin}/program", "do", "something"`.
    system "false"
  end
end
