import './style.css'
import typescriptLogo from './typescript.svg'
import viteLogo from '/vite.svg'
import { setupCounter } from './counter.ts'
import { ArrayProblems } from './arrays/arrays.ts'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div>
    <a href="https://vite.dev" target="_blank">
      <img src="${viteLogo}" class="logo" alt="Vite logo" />
    </a>
    <a href="https://www.typescriptlang.org/" target="_blank">
      <img src="${typescriptLogo}" class="logo vanilla" alt="TypeScript logo" />
    </a>
    <h1>Vite + TypeScript</h1>
    <div class="card">
      <button id="counter" type="button"></button>
    </div>
    <p class="read-the-docs">
      Click on the Vite and TypeScript logos to learn more
    </p>
  </div>
`

setupCounter(document.querySelector<HTMLButtonElement>('#counter')!)

const ArrayProbs = new ArrayProblems('jerk')
ArrayProbs.yolo()

console.time("Function containsDuplicateSort Time"); // Start timer with a label
console.log(ArrayProbs.containsDuplicateSort([1,2,3,4, 5,6,7,8,9,10,11,11]))
console.timeEnd("Function containsDuplicateSort Time"); // Stop timer and print the duration

console.time("Function containsDuplicateHashSet Time"); // Start timer with a label
console.log(ArrayProbs.containsDuplicateHashSet([1,2,3,4, 5,6,7,8,9,10,11,11]))
console.timeEnd("Function containsDuplicateHashSet Time"); // Stop timer and print the duration
