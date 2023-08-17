const defaultTheme = require('tailwindcss/defaultTheme')

module.exports = {
    content: process.env.NODE_ENV == 'production' ? ["./public/js/main.js"] : ["./src/main/**/*.cljs"/*, "./www/js/cljs-runtime/*.js"*/],
    theme: {
        extend: {
            fontFamily: {
                sans: ["Inter var", ...defaultTheme.fontFamily.sans],
            },
        },

        /* These basically behave like the h1, h2, h3, etc. tags in HTML */
        fontSize: {
            'sm': ['1.0rem', {
                lineHeight: '1.5rem',
                letterSpacing: '0.00em',
                fontWeight: '400',
            }],
            'xl': ['1.25rem', {
                lineHeight: '1.75rem',
                letterSpacing: '-0.01em',
                fontWeight: '400',
            }],
            '2xl': ['1.5rem', {
                lineHeight: '2rem',
                letterSpacing: '-0.01em',
                fontWeight: '500',
            }],
            '3xl': ['1.875rem', {
                 lineHeight: '2.25rem',
                 letterSpacing: '-0.02em',
                 fontWeight: '700',
            }],
            '4xl': ['2.0rem', {
                lineHeight: '2.5rem',
                letterSpacing: '-0.02em',
                fontWeight: '900',
            }],
            '5xl': ['3rem', {
                lineHeight: '3rem',
                letterSpacing: '-0.02em',
                fontWeight: '1200',
            }],
          },
    },
    plugins: [
        require('@tailwindcss/forms'),
    ],
}
