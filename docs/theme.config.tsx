import React from 'react'
import { useRouter } from 'next/router'

export default {
  logo: <span>OpenLLM</span>,
  project: {
    link: 'https://github.com/bentoml/OpenLLM',
  },
  chat: {
    link: 'https://l.bentoml.com/join-openllm-discord',
  },
  docsRepositoryBase: 'https://github.com/bentoml/OpenLLM/tree/main/docs',
  footer: {
    text: `OpenLLM, by BentoML Team © ${new Date().getFullYear()}`
  },
  useNextSeoProps() {
    const { asPath } = useRouter()
    if (asPath !== '/') {
      return {
        titleTemplate: '%s – SWR'
      }
    }
  },
}
