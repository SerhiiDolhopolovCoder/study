import re

import pandas as pd
from features.feature_builder import FeatureBuilder


class TitanicFeatureBuilder(FeatureBuilder):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def build_by_template(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """To build features using default template

        Args:
            df (pd.DataFrame): started dataframe | If None -> df from __init__

        Returns:
            pd.DataFrame: dataframe with feuatures
        """
        if df:
            self.df = df
        self.add_missing() \
            .sex_to_male() \
            .fill_embarked() \
            .encode_embarked() \
            .add_status_by_name() \
            .add_role_by_status() \
            .encode_role() \
            .fill_age() \
            .split_cabins() \
            .fill_decks() \
            .encode_decks() \
            .add_bins_age(drop_old=True) \
            .add_bins_fare(drop_old=True) \
            .normalize_age() \
            .normalize_fare() \
            .add_family_size() \
            .split_family_size() \
            .encode_sibSp() \
            .parch_to_child_parent() \
            .drop_irrelevant()
        return self.df

    def sex_to_male(self) -> 'TitanicFeatureBuilder':
        """Convert sex to binary values.
        'male' = 1
        'female' = 0
        OTHERS = -1
        """
        column = 'Sex'
        new_column = 'Male'

        self._check_columns_in_df(column)

        self.df[new_column] = self.df[column].apply(
            lambda x: (1 if x == 'male' else
                       0 if x == 'female' else
                       -1))
        self.df.drop(columns=[column], inplace=True, errors='ignore')

        return self

    def fill_embarked(self) -> 'TitanicFeatureBuilder':
        column = 'Embarked'

        popular_idx = self.df[column].value_counts().idxmax()
        self.df[column] = self.df[column].fillna(popular_idx)
        return self

    def encode_embarked(self) -> 'TitanicFeatureBuilder':
        """Encode embarked column by hot-encoding:

        NOT DUMMY TRAP: FIRST COLUMN WILL BE REMOVED
        """
        return self._hot_encode('Embarked', drop_first=True)

    def add_status_by_name(self) -> 'TitanicFeatureBuilder':
        column = 'Name'
        new_column = 'Status'
        pattern = r'(\w+), (\w+)\.(.*)'
        countess_pattern = 'Countess'

        self._check_columns_in_df(column)

        self.df[new_column] = self.df[column].apply(lambda x: re.search(pattern, x).group(2)
                                                    if re.search(pattern, x)
                                                    else countess_pattern
                                                    if countess_pattern in x
                                                    else None)
        return self

    def add_role_by_status(self) -> 'TitanicFeatureBuilder':
        column = 'Status'
        new_column = 'Role'

        self._check_columns_in_df(column)

        status_role_dict = {
            'Mr': 'civil',
            'Miss': 'civil',
            'Mrs': 'civil',
            'Master': 'kid',
            'Dr': 'civil',
            'Rev': 'elite',
            'Col': 'special',
            'Mlle': 'elite',
            'Major': 'special',
            'Ms': 'elite',
            'Mme': 'elite',
            'Don': 'elite',
            'Lady': 'elite',
            'Sir': 'elite',
            'Capt': 'special',
            'Countess': 'elite',
            'Jonkheer': 'special'
        }

        self.df[new_column] = self.df[column].apply(
            lambda x: status_role_dict[x] if x in status_role_dict else 'other')

        return self

    def encode_role(self) -> 'TitanicFeatureBuilder':
        """Encode role column by hot-encoding:

        NOT DUMMY TRAP: FIRST COLUMN WILL BE REMOVED
        """
        return self._hot_encode('Role', drop_first=True)

    def __get_median_age_by_status(self) -> dict[str, int]:
        """Get median age by status.

        Returns:
            dict[str, int]: status: median age
        """
        column = 'Status'

        self._check_columns_in_df(column)

        return self.df.groupby([column]) \
                      .agg(mean_age=('Age', 'median')) \
                      .to_dict()['mean_age']

    def fill_age(self) -> 'TitanicFeatureBuilder':
        """Fill null ages with median age by status."""
        column = 'Age'
        column_status = 'Status'
        statuses = self.__get_median_age_by_status()
        null_ages = self.df[column].isnull()

        self._check_columns_in_df(column)
        self._check_columns_in_df(column_status)

        self.df.loc[null_ages, column] = self.df[column_status].apply(
            lambda x: statuses[x] if x in statuses else None)

        return self

    def add_bins_age(self, n_bins: int = 10, drop_old: bool = False) -> 'TitanicFeatureBuilder':
        """Create age_bins column from age

        Args:
            n_bins (int): count of bins
        """
        return self._add_bins('Age', n_bins=n_bins, drop_old=drop_old)

    def normalize_age(self) -> 'TitanicFeatureBuilder':
        return self._normalization_0_to_1('Age')

    def split_cabins(self) -> 'TitanicFeatureBuilder':
        """Split cabin
        Cabin -> Deck, Room
        """
        column = 'Cabin'
        column_deck = 'Deck'
        column_room = 'Room'

        self._check_columns_in_df(column)

        self.df[column_deck] = self.df[column].str.slice(0, 1)
        self.df[column_room] = self.df[column].str.extract(
            "([0-9]+)", expand=False)

        return self

    def fill_decks(self) -> 'TitanicFeatureBuilder':
        """Fill missing and 'T' values in Deck column"""
        column = 'Deck'

        self._check_columns_in_df(column)

        self.df[column] = self.df[column].fillna('unknown')

        self.__fill_bad_decks()

        return self

    def __fill_bad_decks(self) -> 'TitanicFeatureBuilder':
        """Fill 'T' values in Deck column"""
        column = 'Deck'

        self._check_columns_in_df(column)

        self.df.loc[self.df['Deck'] == 'T', 'Deck'] = 'unknown'

        return self

    def encode_decks(self) -> 'TitanicFeatureBuilder':
        """Encode deck column by hot-encoding:

        NOT DUMMY TRAP: FIRST COLUMN WILL BE REMOVED
        """
        return self._hot_encode('Deck', drop_first=True)

    def add_bins_fare(self, n_bins: int = 10, drop_old: bool = False) -> 'TitanicFeatureBuilder':
        """Create age_fare column from fare

        Args:
            n_bins (int): count of bins
        """
        return self._add_bins('Fare', n_bins=n_bins, drop_old=drop_old)

    def normalize_fare(self) -> 'TitanicFeatureBuilder':
        return self._normalization_0_to_1('Fare')

    def add_family_size(self) -> 'TitanicFeatureBuilder':
        """Add SibSp and Parch"""
        columnA = 'SibSp'
        columnB = 'Parch'
        new_colum = 'FamilySize'

        self._check_columns_in_df(columnA)
        self._check_columns_in_df(columnB)

        self.df[new_colum] = self.df[columnA] + self.df[columnB]

        return self

    def split_family_size(self) -> 'TitanicFeatureBuilder':
        """Split Family Size -> FamilySize_Alone(removed), FamilySize_Small, FamilySize_Big

        Alone: 0 (myself)
        Small: 1-2 members
        Big: 3+ members


        NOT DUMMY TRAP: Alone COLUMN WILL BE REMOVED
        """
        column = 'FamilySize'
        columnSmall = 'FamilySize_Small'
        columnBig = 'FamilySize_Big'

        self._check_columns_in_df(column)

        self.df[columnSmall] = self.df[column].apply(
            lambda x: (1 if 1 <= x <= 2 else 0))
        self.df[columnBig] = self.df[column].apply(
            lambda x: (1 if x >= 3 else 0))

        self.df.drop(columns=column, inplace=True)

        return self

    def encode_sibSp(self) -> 'TitanicFeatureBuilder':
        """Encode sibSp column by hot-encoding:

        More than 4 will be added to 4
        NOT DUMMY TRAP: FIRST COLUMN WILL BE REMOVED
        """
        column = 'SibSp'
        self._check_columns_in_df(column)

        self.df.loc[self.df[column] > 4, column] = 4
        return self._hot_encode(column, drop_first=True)

    def parch_to_child_parent(self) -> 'TitanicFeatureBuilder':
        """Convert parch to binary values.
        'with child/parent' = 1
        'without' = 0
        OTHERS = -1
        """
        column = 'Parch'
        new_column = 'Child_Parent'

        self._check_columns_in_df(column)

        self.df[new_column] = self.df[column].apply(
            lambda x: (1 if x > 0 else
                       0 if x == 0 else
                       -1))
        self.df.drop(columns=[column], inplace=True, errors='ignore')

        return self

    def drop_irrelevant(self) -> 'TitanicFeatureBuilder':
        # correlation
        self._drop_irrelevant(
            ['Cabin_missing', 'Pclass', 'Age', 'Fare', 'SibSp_1'])
        return self._drop_irrelevant(['PassengerId', 'Name',
                                     'Ticket', 'Cabin', 'Room', 'Status'])
