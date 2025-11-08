import { describe, it, expect } from 'vitest';
import { ArrayProblems } from './arrays';

// 1. Describe a group of related tests (a test suite)
describe('TwoSum Class', () => {
    // Instantiate your class once per suite or per test if needed
    const arrayProblems = new ArrayProblems('yo');

    // 2. Define an individual test case
    it('should return true as 11 is a duplicate', () => {
        const nums = [2, 7, 11, 11, 15];
        
        // 3. Use the expect function to assert the result
        // .toEqual() is used for comparing objects/arrays
        expect(arrayProblems.containsDuplicateSort(nums)).toEqual(true);
    });

    // 2. Define an individual test case
    it('should return false as there are no duplicates', () => {
        const nums = [2, 7, 11, 15];
        
        // 3. Use the expect function to assert the result
        // .toEqual() is used for comparing objects/arrays
        expect(arrayProblems.containsDuplicateSort(nums)).toEqual(false);
    });

});