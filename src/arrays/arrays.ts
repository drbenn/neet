export class ArrayProblems {
  private name: string

  constructor(name: string) {
    this.name = name
  }

  public yolo() {
    console.log('YOLO');
    
  }

  containsDuplicateSort(input: number[]): boolean {

    // sort is better than brute force of taking an unsorted listed and for each item checking every other item...which would be 
    // time complexity O(n^2)
    // space complexity 0(1)

    // with sort we upgrade to
    // time complexity O(nlogn) because the sort algorithm does take some time
    // space complexity 0(1)

    // however, the most optimal solution is using a hash set ... in the next function
    // time complexity O(n)
    // space complexity 0(n)

    const sortedInput: number[] = input.sort((a,b) => a- b)

    let recent: number | null = null

    for (let i = 0; i < sortedInput.length; i++) {
      if (recent === sortedInput[i]) {
        console.log('returning true');
        
        return true
      }
      else recent = sortedInput[i]
    }
    return false
  }

  containsDuplicateHashSet(input: number[]): boolean {
    // however, the most optimal solution is using a hash set
    // time complexity O(n)
    // space complexity 0(n)

    const hashSet: Set<Number> = new Set()

    for (let i = 0; i < input.length; i++) {
      if (hashSet.has(input[i])) {
        return true
      } else {
        hashSet.add(input[i])
      }
    }
    return false
  }
}