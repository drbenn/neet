export class ArrayProblems {
  private name: string

  constructor(name: string) {
    this.name = name
  }

  public yolo() {
    console.log('YOLO');
    
  }

  containsDuplicateSort(input: number[]): boolean {

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

  isAnagram(s: string, t: string): boolean {
    if (s.length !== t.length) {
      return false
    }
    let sList: { [key: string]: number} = {}
    let tList: { [key: string]: number} = {}

    for (let i = 0; i <= s.length; i++) {
      console.log('IUUSHIHU');
      
      const sLetter = s[i]
      console.log(sLetter);
      
      sLetter in sList ? sList[sLetter] += 1 : sList[sLetter] = 1
      console.log(sList);
      

      const tLetter = t[i]
      tLetter in tList ? tList[tLetter] += 1 : tList[tLetter] = 1
    }

    for (const key in sList) {
      if (sList[key] !== tList[key]) {
        return false
      }
    }
            console.log(sList)
        console.log(tList)

    return true
  }
}