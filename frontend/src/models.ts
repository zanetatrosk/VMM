export interface Model {
    label: string;
    value: string;
    breeds: string[];
}

export function parseBreed(breed: string): string {
    return breed.replace(/_/g, ' ');
}

const models: Model[] = [
    {
        label: '16 breeds model',
        value: 'dog',
        breeds: [
            'beagle',
            'boxer',
            'bulldog',
            'chihuahua',
            'corgi',
            'dachshund',
            'german_shepherd',
            'golden_retriever',
            'husky',
            'labrador',
            'pomeranian',
            'poodle',
            'pug',
            'rottweiler',
            'shiba_inu',
            'yorkshire_terrier'
        ]
    },
    {
        label: 'Dog_v2',
        value: 'dog_v2',
        breeds: [
            'affenpinscher',
            'afghan_hound',
            'african_hunting_dog',
            'airedale',
            'american_staffordshire_terrier',
            'appenzeller',
            'australian_terrier',
            'basenji',
            'basset',
            'beagle',
            'bedlington_terrier',
            'bernese_mountain_dog',
            'black-and-tan_coonhound',
            'blenheim_spaniel',
            'bloodhound',
            'bluetick',
            'border_collie',
            'border_terrier',
            'borzoi',
            'boston_bull',
            'bouvier_des_flandres',
            'boxer',
            'brabancon_griffon',
            'briard',
            'brittany_spaniel',
            'bull_mastiff',
            'cairn',
            'cardigan',
            'chesapeake_bay_retriever',
            'chihuahua',
            'chow',
            'clumber',
            'cocker_spaniel',
            'collie',
            'curly-coated_retriever',
            'dandie_dinmont',
            'dhole',
            'dingo',
            'doberman',
            'english_foxhound',
            'english_setter',
            'english_springer',
            'entlebucher',
            'eskimo_dog',
            'flat-coated_retriever',
            'french_bulldog',
            'german_shepherd',
            'german_short-haired_pointer',
            'giant_schnauzer',
            'golden_retriever',
            'gordon_setter',
            'great_dane',
            'great_pyrenees',
            'greater_swiss_mountain_dog',
            'groenendael',
            'ibizan_hound',
            'irish_setter'
        ]    
    }
];

export default models;