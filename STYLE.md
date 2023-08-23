## the coding style

This documentation serves as a brief discussion of the coding style used for
OpenLLM. As you have noticed, it is different from the conventional
[PEP8](https://peps.python.org/pep-0008/) style as across many Python projects.
The manifestation of OpenLLM code style is a combination of
[Google Python Style](https://google.github.io/styleguide/pyguide.html),
inspiration from coding language such as APL, Haskell, and is designed for fast,
experimental development and prototyping.

> [!NOTE]
> Some of this style can also be applied to TS/JS within the monorepo, and
> I have setup tools to make sure style is consistent across languages.

Everyone always has their own opinions on style. I believe this is exemplified
further within the Python community, as it tries to be beginner-friendly, and
therefore most people hold a very strong opinion on styling. I don't have a
strong opinion on style either (I don't have any issue with PEP8, as we use it
for our other projects), as long as:

- You don't use any linter, formatter that change the style drastically other
  than what specified within the projects' [`pyproject.toml`](./pyproject.toml).
- The code you contribute is not widely different from the style of the code
  surrounding it.

With that being said, I want to use this project as a playground the explore a
style that is both: "feels natural" and expressive for mathematical reasoning. I
hope that you find this guide somewhat thought-provoking and interesting, that
you can iterate and try to adopt some of them as part of the process
contributing to the library.

While PEP8 is a great base for a style guide, I find it to be having way too
much white spaces and makes the code feels 'robotic'. Having a deterministic
style and formatter is great to reduce the overhead of stylistic discussions,
but I think it is important to write code that express the intent of reasoning.
(_The policy here is definitely not "shovel everything into one line", but
rather "compact and flowing"_)

The styling is heavily inspired by
[Kenneth Iverson's](https://en.wikipedia.org/wiki/Kenneth_E._Iverson) 1979
Turing award lecture,
[Notation as a Tool of Thought](https://www.eecg.toronto.edu/~jzhu/csc326/readings/iverson.pdf),
and a lot of the stylistic inspiration comes from
[Jeremy Howard's](https://jeremy.fast.ai/) [fastai](https://docs.fast.ai/). One
thing that has been stuck with me ever since is the idea of "brevity facilitates
reasoning", as such the tersity of style aren't just for the sake of shortness,
rather the brevity of expression. (it enables
[expository programming](http://archive.vector.org.uk/art10000980), combining
with prototyping new ideas and logics within models implementation)

## some guidelines

Though I have stopped using deterministic formatter and linter, I do understand
that people have preferences for using these tools, and it plays nicely with IDE
and editors. As such, I included a [`pyproject.toml`](./pyproject.toml) file
that specifies some configuration for the tools that makes it compiliant with
the repository's style. In short, some of the tools include `ruff`, `yapf`, and
`interrogate`. Since we manage everything via `hatch`, refer back to the
[DEVELOPMENT.md](./DEVELOPMENT.md) for more information on this.

Overtime, Python has incorporated a lot of features that supports this style of
coding, including list comprehension, generator expression, lambda, array-based
programming. Yet, Python will remain verbose per se, and the goal is that to
make code fit nicely on a screen, and we don't have to always scroll downwards.

While brevity is important, it is also important to make sure functions are
somewhat, type-safe. Since there is no real type-safety when working with
Python, typing should be a best-effort to make sure we don't introduce too many
bugs.

### naming

- follow Python standard for this, I don't have too much opinion on this. Just
  make sure that it is descriptive, and the abbreviation describes the intent of
  the variable. i.e: `to_gpu` instead of `t_gpu`, `to_cpu` instead of `t_cpu`.
- any math-related notation or neural net layers should be expressive and stay
  close to the paper as much as possible. For example: `lm_head.weight` instead
  of `lm_head.w`. Espically for implementing custom kernels and layers, it is
  crucial to follow its nomenclature. E.g: `conv1` instead of
  `first_conv_layer`.
- for functions, try to use verb-noun naming convention. i.e: `get_tokenizer`,
- also just use single quotes for string, and double quotes for within string when needed.
  i.e: `f'hello "{name}"'`

_If you have any suggestions, feel free to give it on our discord server!_

### layout

- Preferably not a lot of whitespaces, but rather flowing. If you can fit
  everything for `if`, `def` or a `return` within one line, then there's no need
  to break it into multiple line:

  ```python
  def foo(x): return rotate_cv(x) if x > 0 else -x
  ```

- imports should be grouped by their types: standard library, third-party, and local

  ```python
  import os, sys
  import orjson, bentoml
  ```
  This is partially to make it easier to work with merge-conflicts, and easier
  for IDE to navigate context definition.

- indent with 2 spaces, which follow the Google codestyle.

- With regards to writing operator, try to follow the domain-specific notation.
  I.e: when writing pathlib, just don't add space since that is not how you
  write a path in the terminal. `yapf` will try to accommodate some of this
  changes.

- Avoid trailing whitespace

- use array, pytorch or numpy-based indexing where possible.

- If you need to export anything, put it in `__all__` or do lazy export for
  type-safe checker.

### misc

- import alias should be concise and descriptive. A convention is to always
  `import typing as t`.
- Writing docstring when it is possible. No need to comment everything asn it
  makes the codebase hard to read. For docstring, follow the Google style guide.
- We do lazy imports, so consult some of the `__init__.py` to see how we do it.
- Documentation is still _working-in-progress_, but tldr it will be written in
  MDX and will be hosted on the GitHub Pages, so stay tuned!
- If anything that is not used for runtime, just put it under `t.TYPE_CHECKING`

### note on codegen

- We also do some codegen for some of the assignment functions. These logics are
  largely based on the work of [attrs](https://github.com/python-attrs/attrs) to
  ensure fast and isolated codegen in Python. If you need codegen but don't know
  how it works, feel free to mention @aarnphm on discord!

## FAQ

### Why not use `black`?

`black` is used on our other projects, but I rather find `black` to be very
verbose and overtime it is annoying to work with too much whitespaces.

### Why not PEP8?

PEP8 is great if you are writing library such as this, but I'm going to do a lot
of experimenting for implementing papers, so I decided early on that PEP8 is
probably not fit here, and want to explore more expressive style.

### Editor is complaining about the style, what should I do?

Kindly ask you to disable linting for this project 🤗. I will try my best to
accomodate with ruff and yapf, but I don't want to spend too much time on this.
It is pretty stragithforward to disable it in your editor, with google.

### Style might put off new contributors?

I don't think so, as mentioned before, I don't have too much opinion on style as
long as it somewhat follow what I have described above or the style of the code
surrounding it. I will still accept styles PR as long as it is not too drastic.
Just make sure to add the revision to `.git-blame-ignore-revs` so that
`git blame` would work correctly.

As for people who are too close-minded about styling, such individuals aren't
the ones we want to work with anyway!
